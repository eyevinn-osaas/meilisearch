mod rebench;

use std::collections::BTreeMap;
use std::fmt::Display;
use std::io::{BufReader, Read, Seek, Write};
use std::path::PathBuf;

use anyhow::{bail, Context};
use clap::Parser;
use futures_util::TryStreamExt;
use serde::Deserialize;
use serde_json::json;
use sha2::Digest;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Layer;
use tracing_trace::processor::span_stats::CallStats;

use self::rebench::Criterion;

pub fn default_http_addr() -> String {
    "127.0.0.1:7700".to_string()
}
pub fn default_report_folder() -> String {
    "./reports/".into()
}

pub fn default_asset_folder() -> String {
    "./assets/".into()
}

pub fn default_log_filter() -> String {
    "info".into()
}

/// Run benchmarks from a workload
#[derive(Parser, Debug)]
pub struct BenchDeriveArgs {
    /// Filename of the workload file, pass multiple filenames
    /// to run multiple workloads in the specified order.
    ///
    /// Each workload run will get its own report file.
    #[arg(value_name = "WORKLOAD_FILE", last = true)]
    workload_file: Vec<PathBuf>,

    /// Directory to output reports.
    #[arg(short, long, default_value_t = default_report_folder())]
    report_folder: String,

    /// Directory to store the remote assets.
    #[arg(short, long, default_value_t = default_asset_folder())]
    asset_folder: String,

    /// Log directives
    #[arg(short, long, default_value_t = default_log_filter())]
    log_filter: String,
}

#[derive(Deserialize)]
pub struct Workload {
    pub name: String,
    pub run_count: u16,
    pub extra_cli_args: Vec<String>,
    pub assets: BTreeMap<String, Asset>,
    pub commands: Vec<Command>,
}

#[derive(Deserialize, Clone)]
pub struct Asset {
    pub local_location: Option<String>,
    pub remote_location: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Clone, Deserialize)]
pub struct Command {
    pub route: String,
    pub method: Method,
    #[serde(default)]
    pub body: Body,
    #[serde(default)]
    pub synchronous: SyncMode,
}

#[derive(Default, Clone, Deserialize)]
#[serde(untagged)]
pub enum Body {
    Inline {
        inline: serde_json::Value,
    },
    Asset {
        asset: String,
    },
    #[default]
    Empty,
}

impl Body {
    pub fn get(
        self,
        assets: &BTreeMap<String, Asset>,
        asset_folder: &str,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        Ok(match self {
            Body::Inline { inline: body } => Some(body),
            Body::Asset { asset: name } => Some({
                let file: std::fs::File = fetch_asset(&name, assets, asset_folder)
                    .with_context(|| format!("while getting body from asset '{name}'"))?;
                serde_json::from_reader(file)
                    .with_context(|| format!("could not deserialize asset '{name}' as JSON"))?
            }),
            Body::Empty => None,
        })
    }
}

fn fetch_asset(
    name: &str,
    assets: &BTreeMap<String, Asset>,
    asset_folder: &str,
) -> anyhow::Result<std::fs::File> {
    let asset =
        assets.get(name).with_context(|| format!("could not find asset with name '{name}'"))?;
    let filename = if let Some(local_filename) = &asset.local_location {
        local_filename.clone()
    } else {
        format!("{asset_folder}/{name}")
    };

    std::fs::File::open(&filename)
        .with_context(|| format!("could not open asset '{name}' at '{filename}'"))
}

impl Display for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {} ({:?})", self.method, self.route, self.synchronous)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum Method {
    GET,
    POST,
    PATCH,
    DELETE,
    PUT,
}

impl From<Method> for reqwest::Method {
    fn from(value: Method) -> Self {
        match value {
            Method::GET => Self::GET,
            Method::POST => Self::POST,
            Method::PATCH => Self::PATCH,
            Method::DELETE => Self::DELETE,
            Method::PUT => Self::PUT,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Deserialize)]
pub enum SyncMode {
    DontWait,
    #[default]
    WaitForResponse,
    WaitForTask,
}

fn new_client(
    master_key: &str,
    timeout: Option<std::time::Duration>,
) -> anyhow::Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.append(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {master_key}"))
            .context("Invalid authorization header")?,
    );
    let client = reqwest::ClientBuilder::new().default_headers(headers);
    let client = if let Some(timeout) = timeout { client.timeout(timeout) } else { client };
    Ok(client.build()?)
}

pub fn run(args: BenchDeriveArgs) -> anyhow::Result<()> {
    let filter: tracing_subscriber::filter::Targets =
        args.log_filter.parse().context("invalid --log-filter")?;

    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer().with_span_events(FmtSpan::ENTER).with_filter(filter),
    );
    tracing::subscriber::set_global_default(subscriber).context("could not setup logging")?;

    let rt = tokio::runtime::Builder::new_current_thread().enable_io().enable_time().build()?;
    let _scope = rt.enter();

    let master_key = meilisearch_auth::generate_master_key();

    let client = new_client(&master_key, Some(std::time::Duration::from_secs(60)))?;

    rt.block_on(async {
        tracing::info!(workload_count = args.workload_file.len(), "handling workload files");
        for workload_file in args.workload_file.iter() {
            let workload: Workload = serde_json::from_reader(
                std::fs::File::open(workload_file)
                    .with_context(|| format!("error opening {}", workload_file.display()))?,
            )
            .with_context(|| format!("error parsing {} as JSON", workload_file.display()))?;

            run_workload(&client, &master_key, workload, &args).await?;
        }
        Ok::<(), anyhow::Error>(())
    })?;

    tracing::info!("Success");

    Ok(())
}

#[tracing::instrument(skip(client, workload, master_key, args), fields(workload = workload.name))]
async fn run_workload(
    client: &reqwest::Client,
    master_key: &str,
    workload: Workload,
    args: &BenchDeriveArgs,
) -> anyhow::Result<()> {
    fetch_assets(client, &workload.assets, &args.asset_folder).await?;

    let mut tasks = Vec::new();

    for i in 0..workload.run_count {
        tasks.push(run_workload_run(client, master_key, &workload, args, i).await?);
    }

    let mut reports = Vec::with_capacity(workload.run_count as usize);

    for task in tasks {
        reports.push(
            task.await
                .context("task panicked while processing report")?
                .context("task failed while processing report")?,
        );
    }

    runs_to_rebench(&workload, client, files_to_callstat(reports)).await?;

    tracing::info!(workload = workload.name, "Successful workload");

    Ok(())
}

fn files_to_callstat(
    reports: Vec<std::fs::File>,
) -> impl Iterator<Item = anyhow::Result<BTreeMap<String, CallStats>>> {
    reports.into_iter().map(|file| {
        let mut map = BTreeMap::new();
        for res in serde_json::Deserializer::from_reader(BufReader::new(file)).into_iter() {
            let value: BTreeMap<String, CallStats> =
                res.context("could not deserialize report file")?;

            map.extend(value.into_iter());
        }
        Ok(map)
    })
}

#[tracing::instrument(skip(client, assets), fields(asset_count = assets.len()))]
async fn fetch_assets(
    client: &reqwest::Client,
    assets: &BTreeMap<String, Asset>,
    asset_folder: &str,
) -> anyhow::Result<()> {
    let mut download_tasks = tokio::task::JoinSet::new();
    for (name, asset) in assets {
        // trying local
        if let Some(local) = &asset.local_location {
            match std::fs::File::open(local) {
                Ok(file) => {
                    if check_sha256(name, asset, file)? {
                        continue;
                    } else {
                        tracing::warn!(asset = name, file = local, "found local resource for asset but hash differed, skipping to asset store");
                    }
                }
                Err(error) => match error.kind() {
                    std::io::ErrorKind::NotFound => { /* file does not exist, go to remote, no need for logs */
                    }
                    _ => tracing::warn!(
                        error = &error as &dyn std::error::Error,
                        "error checking local resource, skipping to asset store"
                    ),
                },
            }
        }

        // checking asset store
        let store_filename = format!("{}/{}", asset_folder, name);

        match std::fs::File::open(&store_filename) {
            Ok(file) => {
                if check_sha256(name, asset, file)? {
                    continue;
                } else {
                    tracing::warn!(asset = name, file = store_filename, "found resource for asset in asset store, but hash differed, skipping to remote method");
                }
            }
            Err(error) => match error.kind() {
                std::io::ErrorKind::NotFound => { /* file does not exist, go to remote, no need for logs */
                }
                _ => tracing::warn!(
                    error = &error as &dyn std::error::Error,
                    "error checking resource in store, skipping to remote method"
                ),
            },
        }

        // downloading remote
        match &asset.remote_location {
            Some(location) => {
                std::fs::create_dir_all(asset_folder).with_context(|| format!("could not create asset folder at {asset_folder}"))?;
                download_tasks.spawn({
                    let client = client.clone();
                    let name = name.to_string();
                    let location = location.to_string();
                    let store_filename = store_filename.clone();
                    let asset = asset.clone();
                    download_asset(client, name, asset, location, store_filename)});
            },
            None => bail!("asset {name} has no remote location, but was not found locally or in the asset store"),
        }
    }

    while let Some(res) = download_tasks.join_next().await {
        res.context("download task panicked")?.context("download task failed")?;
    }

    Ok(())
}

fn check_sha256(name: &str, asset: &Asset, mut file: std::fs::File) -> anyhow::Result<bool> {
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).with_context(|| format!("hashing file for asset {name}"))?;
    let mut file_hash = sha2::Sha256::new();
    file_hash.update(&bytes);
    let file_hash = file_hash.finalize();
    let file_hash = format!("{:x}", file_hash);
    tracing::debug!(hash = file_hash, "hashed local file");

    Ok(match &asset.sha256 {
        Some(hash) => {
            tracing::debug!(hash, "hash from workload");
            hash.to_ascii_lowercase() == file_hash
        }
        None => {
            tracing::warn!(sha256 = file_hash, "Skipping hash for asset {name} that doesn't have one. Please add it to workload file");
            true
        }
    })
}

#[tracing::instrument(skip(client, asset, name), fields(asset = name))]
async fn download_asset(
    client: reqwest::Client,
    name: String,
    asset: Asset,
    src: String,
    dest_filename: String,
) -> anyhow::Result<()> {
    let context = || format!("failure downloading asset {name} from {src}");

    let response = client.get(&src).send().await.with_context(context)?;

    let dest = std::fs::File::create(&dest_filename)
        .with_context(|| format!("creating destination file {dest_filename}"))
        .with_context(context)?;

    let mut dest = std::io::BufWriter::new(dest);

    let writing_context = || format!("while writing to destination file at {dest_filename}");

    let mut response = response.bytes_stream();

    while let Some(bytes) =
        response.try_next().await.context("while downloading file").with_context(context)?
    {
        dest.write_all(&bytes).with_context(writing_context).with_context(context)?;
    }

    let file = dest.into_inner().with_context(writing_context).with_context(context)?;

    if !check_sha256(&name, &asset, file)? {
        bail!("asset '{name}': sha256 mismatch for file {dest_filename} downloaded from {src}")
    }

    Ok(())
}

#[tracing::instrument(skip(client, workload, master_key, args), fields(workload = %workload.name))]
async fn run_workload_run(
    client: &reqwest::Client,
    master_key: &str,
    workload: &Workload,
    args: &BenchDeriveArgs,
    run_number: u16,
) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<std::fs::File>>> {
    delete_db();
    build_meilisearch().await?;
    let meilisearch = start_meilisearch(client, master_key, workload, &args.asset_folder).await?;
    let processor = run_commands(client, master_key, workload, args, run_number).await?;

    kill_meilisearch(meilisearch).await;

    tracing::info!(run_number, "Successful run");

    Ok(processor)
}

async fn kill_meilisearch(mut meilisearch: tokio::process::Child) {
    if let Err(error) = meilisearch.kill().await {
        tracing::warn!(
            error = &error as &dyn std::error::Error,
            "while terminating Meilisearch server"
        )
    }
}

#[tracing::instrument]
async fn build_meilisearch() -> anyhow::Result<()> {
    let mut command = tokio::process::Command::new("cargo");
    command.arg("build").arg("--release").arg("-p").arg("meilisearch");

    command.kill_on_drop(true);

    let mut builder = command.spawn().context("error building Meilisearch")?;

    if !builder.wait().await.context("could not build Meilisearch")?.success() {
        bail!("failed building Meilisearch")
    }

    Ok(())
}

#[tracing::instrument(skip(client, master_key, workload), fields(workload = workload.name))]
async fn start_meilisearch(
    client: &reqwest::Client,
    master_key: &str,
    workload: &Workload,
    asset_folder: &str,
) -> anyhow::Result<tokio::process::Child> {
    let mut command = tokio::process::Command::new("cargo");
    command
        .arg("run")
        .arg("--release")
        .arg("-p")
        .arg("meilisearch")
        .arg("--bin")
        .arg("meilisearch")
        .arg("--");

    command.arg("--db-path").arg("./_xtask_benchmark.ms");
    command.arg("--master-key").arg(master_key);

    for extra_arg in workload.extra_cli_args.iter() {
        command.arg(extra_arg);
    }

    command.kill_on_drop(true);

    let mut meilisearch = command.spawn().context("Error starting Meilisearch")?;

    wait_for_health(client, &mut meilisearch, &workload.assets, asset_folder).await?;

    Ok(meilisearch)
}

async fn wait_for_health(
    client: &reqwest::Client,
    meilisearch: &mut tokio::process::Child,
    assets: &BTreeMap<String, Asset>,
    asset_folder: &str,
) -> anyhow::Result<()> {
    for i in 0..100 {
        let res = run_command(client.clone(), health_command(), assets, asset_folder).await;
        if res.is_ok() {
            // check that this is actually the current Meilisearch instance that answered us
            if let Some(exit_code) =
                meilisearch.try_wait().context("cannot check Meilisearch server process status")?
            {
                tracing::error!("Got an health response from a different process");
                bail!("Meilisearch server exited early with code {exit_code}");
            }

            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        // check whether the Meilisearch instance exited early (cut the wait)
        if let Some(exit_code) =
            meilisearch.try_wait().context("cannot check Meilisearch server process status")?
        {
            bail!("Meilisearch server exited early with code {exit_code}");
        }
        tracing::debug!(attempt = i, "Waiting for Meilisearch to go up");
    }
    bail!("meilisearch is not responding")
}

fn health_command() -> Command {
    Command {
        route: "/health".into(),
        method: Method::GET,
        body: Default::default(),
        synchronous: SyncMode::WaitForResponse,
    }
}

fn delete_db() {
    let _ = std::fs::remove_dir_all("./_xtask_benchmark.ms");
}

async fn run_commands(
    client: &reqwest::Client,
    master_key: &str,
    workload: &Workload,
    args: &BenchDeriveArgs,
    run_number: u16,
) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<std::fs::File>>> {
    let report_folder = &args.report_folder;
    let workload_name = &workload.name;

    std::fs::create_dir_all(report_folder)
        .with_context(|| format!("could not create report directory at {report_folder}"))?;

    let trace_filename = format!("{report_folder}/{workload_name}-{run_number}-trace.json");
    let report_filename = format!("{report_folder}/{workload_name}-{run_number}-report.json");

    let report_handle = start_report(master_key, trace_filename).await?;

    for batch in workload
        .commands
        .as_slice()
        .split_inclusive(|command| !matches!(command.synchronous, SyncMode::DontWait))
    {
        run_batch(client, batch, &workload.assets, &args.asset_folder).await?;
    }

    let processor = stop_report(client, report_filename, report_handle).await?;

    Ok(processor)
}

async fn stop_report(
    client: &reqwest::Client,
    filename: String,
    report_handle: tokio::task::JoinHandle<anyhow::Result<std::fs::File>>,
) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<std::fs::File>>> {
    let response = client.delete(url_of("logs")).send().await.context("while stopping report")?;
    if !response.status().is_success() {
        bail!("received HTTP {} while stopping report", response.status())
    }

    let mut file = tokio::time::timeout(std::time::Duration::from_secs(1000), report_handle)
        .await
        .context("while waiting for the end of the report")?
        .context("report writing task panicked")?
        .context("while writing report")?;

    file.rewind().context("while rewinding report file")?;

    let process_handle = tokio::task::spawn_blocking(move || -> anyhow::Result<std::fs::File> {
        let span = tracing::info_span!("processing trace to report", filename);
        let _guard = span.enter();
        let report = tracing_trace::processor::span_stats::to_call_stats(
            tracing_trace::TraceReader::new(std::io::BufReader::new(file)),
        )
        .context("could not convert trace to report")?;
        let context = || format!("writing report to {filename}");

        let mut output_file = std::io::BufWriter::new(
            std::fs::File::options()
                .create(true)
                .truncate(true)
                .write(true)
                .read(true)
                .open(&filename)
                .with_context(context)?,
        );

        for (key, value) in report {
            serde_json::to_writer(&mut output_file, &json!({key: value}))
                .context("serializing span stat")?;
            writeln!(&mut output_file).with_context(context)?;
        }
        output_file.flush().with_context(context)?;
        let mut output_file = output_file.into_inner().with_context(context)?;

        output_file.rewind().context("could not rewind ouptut_file").with_context(context)?;

        tracing::info!("success");
        Ok(output_file)
    });

    Ok(process_handle)
}

async fn runs_to_rebench<'a>(
    workload: &Workload,
    client: &reqwest::Client,
    reports: impl Iterator<Item = anyhow::Result<BTreeMap<String, CallStats>>>,
) -> anyhow::Result<()> {
    let environment = rebench::Environment::generate_from_current_config();

    let (source, time) =
        rebench::Source::from_repo(".").context("while getting source repository information")?;

    let mut benchmark_data = rebench::BenchmarkData::new(environment, source, &workload.name, time);
    benchmark_data.with_project("Meilisearch");
    let mut benchmarks = BTreeMap::new();

    benchmark_data.push_criterion(Criterion { id: 0, name: "time".into(), unit: "ns".into() });
    benchmark_data.push_criterion(Criterion { id: 1, name: "calls".into(), unit: "".into() });

    for (run_index, report) in reports.enumerate() {
        let report = report?;
        for (span, stats) in report {
            let benchmark = benchmarks.entry(span.clone()).or_insert(rebench::Benchmark {
                name: span.clone(),
                suite: rebench::Suite {
                    name: "Meilisearch".into(),
                    desc: None,
                    executor: rebench::Executor { name: "Meilisearch".into(), desc: None },
                },
                run_details: rebench::RunDetails {
                    max_invocation_time: 0,
                    min_iteration_time: 0,
                    warmup: None,
                },
                desc: None,
            });

            let run_id = rebench::RunId {
                benchmark: benchmark.clone(),
                cmdline: Default::default(),
                location: Default::default(),
                var_value: None,
                cores: None,
                input_size: None,
                extra_args: None,
            };

            let mut run = rebench::Run::new(run_id);

            let mut point = rebench::DataPoint::new(run_index, 0);
            point.add_point(rebench::Measure { criterion_id: 0, value: stats.ns as f64 });
            point.add_point(rebench::Measure { criterion_id: 1, value: stats.nb as f64 });
            run.add_data(point);
            benchmark_data.push_run(run);
        }
    }

    /// FIXME: fetch rebenchdb url
    let response = client
        .put("http://localhost:33333/rebenchdb/results")
        .json(&benchmark_data)
        .send()
        .await
        .context("could not send data to rebenchdb")?;
    if !response.status().is_success() {
        bail!(
            "sending results to rebenchdb failed with HTTP {}: {}",
            response.status(),
            response.text().await.unwrap_or("unknown".to_string())
        )
    }

    Ok(())
}

async fn start_report(
    master_key: &str,
    filename: String,
) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<std::fs::File>>> {
    let report_file = std::fs::File::options()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(&filename)
        .with_context(|| format!("could not create file at {filename}"))?;
    let mut report_file = std::io::BufWriter::new(report_file);

    // reporting uses its own client because keeping the stream open to wait for entries
    // blocks any other requests
    // Also we don't want any pesky timeout because we don't know how much time it will take to recover the full trace
    let client = new_client(master_key, None)?;

    let response = client
        .post(url_of("logs"))
        .json(&json!({
            "mode": "profile",
            "target": "indexing::=trace"
        }))
        .send()
        .await
        .context("failed to start report")?;

    let code = response.status();
    if code.is_client_error() {
        tracing::error!(%code, "request error when trying to start report");
        let response: serde_json::Value =
            response.json().await.context("could not deserialize response as JSON")?;
        bail!(
            "request error when trying to start report: server responded with error code {code} and '{response}'"
        )
    } else if code.is_server_error() {
        tracing::error!(%code, "server error when trying to start report");
        let response: serde_json::Value =
            response.json().await.context("could not deserialize response as JSON")?;
        bail!("server error when trying to start report: server responded with error code {code} and '{response}'")
    }

    Ok(tokio::task::spawn(async move {
        let mut stream = response.bytes_stream();
        while let Some(bytes) = stream.try_next().await.context("while waiting for report")? {
            report_file
                .write_all(&bytes)
                .with_context(|| format!("while writing report to {filename}"))?;
        }
        report_file.into_inner().with_context(|| format!("while writing report to {filename}"))
    }))
}

async fn run_batch(
    client: &reqwest::Client,
    batch: &[Command],
    assets: &BTreeMap<String, Asset>,
    asset_folder: &str,
) -> anyhow::Result<()> {
    let [.., last] = batch else { return Ok(()) };
    let sync = last.synchronous;

    let mut tasks = tokio::task::JoinSet::new();

    for command in batch {
        // FIXME: you probably don't want to copy assets everytime here
        tasks.spawn({
            let client = client.clone();
            let command = command.clone();
            let assets = assets.clone();
            let asset_folder = asset_folder.to_owned();

            async move { run_command(client, command, &assets, &asset_folder).await }
        });
    }

    while let Some(result) = tasks.join_next().await {
        result
            .context("panicked while executing command")?
            .context("error while executing command")?;
    }

    match sync {
        SyncMode::DontWait => {}
        SyncMode::WaitForResponse => {}
        SyncMode::WaitForTask => wait_for_tasks(client).await?,
    }

    Ok(())
}

async fn wait_for_tasks(client: &reqwest::Client) -> anyhow::Result<()> {
    loop {
        let response = client
            .get(url_of("tasks?statuses=enqueued,processing"))
            .send()
            .await
            .context("Could not wait for tasks")?;
        let response: serde_json::Value =
            response.json().await.context("Could not deserialize response to JSON")?;
        match response.get("total") {
            Some(serde_json::Value::Number(number)) => {
                let number = number.as_u64().with_context(|| {
                    format!("waiting for tasks: could not parse 'total' as integer, got {}", number)
                })?;
                if number == 0 {
                    break;
                } else {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
            }
            Some(thing_else) => {
                bail!(format!(
                    "waiting for tasks: could not parse 'total' as a number, got '{thing_else}'"
                ))
            }
            None => {
                bail!(format!(
                    "waiting for tasks: expected response to contain 'total', got '{response}'"
                ))
            }
        }
    }
    Ok(())
}

#[tracing::instrument(skip(client, command, assets, asset_folder), fields(command = %command))]
async fn run_command(
    client: reqwest::Client,
    mut command: Command,
    assets: &BTreeMap<String, Asset>,
    asset_folder: &str,
) -> anyhow::Result<()> {
    // memtake the body here to leave an empty body in its place, so that command is not partially moved-out
    let body = std::mem::take(&mut command.body)
        .get(assets, asset_folder)
        .with_context(|| format!("while getting body for command {command}"))?;

    let response = client
        .request(command.method.into(), url_of(&command.route))
        .json(&body)
        .send()
        .await
        .with_context(|| format!("error sending command: {}", command))?;

    let code = response.status();
    if code.is_client_error() {
        tracing::error!(%command, %code, "error in workload file");
        let response: serde_json::Value =
            response.json().await.context("Could not deserialize response as JSON")?;
        bail!("error in workload file: server responded with error code {code} and '{response}'")
    } else if code.is_server_error() {
        tracing::error!(%command, %code, "server error");
        let response: serde_json::Value =
            response.json().await.context("Could not deserialize response as JSON")?;
        bail!("server error: server responded with error code {code} and '{response}'")
    }

    Ok(())
}

fn url_of(route: &str) -> String {
    format!("http://127.0.0.1:7700/{route}")
}
