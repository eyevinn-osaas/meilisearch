use std::collections::BTreeMap;
use std::fmt::Display;
use std::io::{Read, Write};
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

pub fn default_http_addr() -> String {
    "127.0.0.1:7700".to_string()
}
pub fn default_report_folder() -> String {
    "./report/".into()
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
pub enum Body {
    Inline(serde_json::Value),
    Asset(String),
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
            Body::Inline(body) => Some(body),
            Body::Asset(name) => Some({
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
}

impl From<Method> for reqwest::Method {
    fn from(value: Method) -> Self {
        match value {
            Method::GET => Self::GET,
            Method::POST => Self::POST,
            Method::PATCH => Self::PATCH,
            Method::DELETE => Self::DELETE,
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

    let mut headers = reqwest::header::HeaderMap::new();
    headers.append(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {master_key}"))
            .context("Invalid authorization header")?,
    );
    let client = reqwest::ClientBuilder::new()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

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

    for i in 0..workload.run_count {
        run_workload_run(client, master_key, &workload, args, i).await?
    }
    Ok(())
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
) -> anyhow::Result<()> {
    delete_db()?;
    build_meilisearch().await?;
    let meilisearch = start_meilisearch(client, master_key, workload, &args.asset_folder).await?;
    run_commands(client, workload, args, run_number).await?;

    kill_meilisearch(meilisearch).await;

    Ok(())
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

fn delete_db() -> anyhow::Result<()> {
    let _ = std::fs::remove_dir_all("./_xtask_benchmark.ms");
    Ok(())
}

async fn run_commands(
    client: &reqwest::Client,
    workload: &Workload,
    args: &BenchDeriveArgs,
    run_number: u16,
) -> anyhow::Result<()> {
    let report_handle =
        start_report(client, &workload.name, &args.report_folder, run_number).await?;

    for batch in workload
        .commands
        .as_slice()
        .split_inclusive(|command| !matches!(command.synchronous, SyncMode::DontWait))
    {
        run_batch(client, batch, &workload.assets, &args.asset_folder).await?;
    }

    stop_report(client, report_handle).await?;

    Ok(())
}

async fn stop_report(
    client: &reqwest::Client,
    report_handle: tokio::task::JoinHandle<Result<(), anyhow::Error>>,
) -> anyhow::Result<()> {
    client.delete(url_of("logs")).send().await.context("while stopping report")?;

    tokio::time::timeout(std::time::Duration::from_secs(60), report_handle)
        .await
        .context("while waiting for the end of the report")?
        .context("report writing task panicked")?
        .context("while writing report")?;

    Ok(())
}

async fn start_report(
    client: &reqwest::Client,
    workload_name: &str,
    report_folder: &str,
    run_number: u16,
) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<()>>> {
    std::fs::create_dir_all(report_folder)
        .with_context(|| format!("could not create report directory at {report_folder}"))?;

    let filename = format!("{report_folder}/{workload_name}-{run_number}-report.json");
    let report_file = std::fs::File::create(&filename)
        .with_context(|| format!("could not create file at {filename}"))?;
    let mut report_file = std::io::BufWriter::new(report_file);

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
        report_file.flush().with_context(|| format!("while writing report to {filename}"))?;
        Ok(())
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
