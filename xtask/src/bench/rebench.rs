use std::path::Path;

use git2::ErrorCode;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

fn none<T>() -> Option<T> {
    None
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkData {
    pub data: Vec<Run>,
    pub criteria: Vec<Criterion>,
    pub env: Environment,
    pub source: Source,

    pub experiment_name: String,
    pub experiment_description: Option<String>,

    #[serde(with = "time::serde::rfc3339")]
    pub start_time: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339::option", default = "none")]
    pub end_time: Option<OffsetDateTime>,
    pub project_name: Option<String>,
}

impl BenchmarkData {
    pub fn new(
        env: Environment,
        source: Source,
        experiment_name: impl AsRef<str>,
        start_time: time::OffsetDateTime,
    ) -> Self {
        Self {
            data: Vec::new(),
            criteria: Vec::new(),
            env,
            source,
            experiment_name: experiment_name.as_ref().to_string(),
            experiment_description: None,
            start_time,
            end_time: None,
            project_name: None,
        }
    }

    pub fn with_project(&mut self, project: impl AsRef<str>) {
        self.project_name = Some(project.as_ref().to_string());
    }

    pub fn push_run(&mut self, run: Run) {
        self.data.push(run);
    }

    pub fn push_criterion(&mut self, criteria: Criterion) {
        self.criteria.push(criteria);
    }

    pub fn end_time(&mut self, end_time: OffsetDateTime) {
        self.end_time = Some(end_time);
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Run {
    /// The different points of your run
    #[serde(rename = "d")]
    pub points: Vec<DataPoint>,
    /// TODO: What is this?
    #[serde(rename = "p", skip_serializing_if = "Vec::is_empty", default = "Vec::new")]
    pub profile: Vec<ProfileData>,
    /// The id of your run
    pub run_id: RunId,
}

impl Run {
    pub fn new(run_id: RunId) -> Self {
        Self { points: Vec::new(), profile: Vec::new(), run_id }
    }

    pub fn add_data(&mut self, data_point: DataPoint) {
        self.points.push(data_point)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Executor {
    /// Name of the executor
    pub name: String,
    /// Description of the executor
    pub desc: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Suite {
    /// Name of the benchmark suite
    pub name: String,
    /// Description of the benchmark suite
    pub desc: Option<String>,
    /// The executor that ran the benchmarks
    pub executor: Executor,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunDetails {
    /// TODO: doc
    pub max_invocation_time: usize,
    /// TODO: doc
    pub min_iteration_time: usize,
    /// The number of warmup runs made before running the actual benchmark
    pub warmup: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Benchmark {
    /// Name of the benchmark
    pub name: String,
    pub suite: Suite,
    pub run_details: RunDetails,
    /// Description of the benchmark
    pub desc: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunId {
    pub benchmark: Benchmark,
    pub cmdline: String,

    /// The current working directory.
    pub location: String,

    pub var_value: Option<String>,
    pub cores: Option<usize>,
    pub input_size: Option<String>,
    pub extra_args: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Measure {
    #[serde(rename = "c")]
    pub criterion_id: usize,

    #[serde(rename = "v")]
    pub value: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DataPoint {
    #[serde(rename = "in")]
    pub invocation: usize,

    #[serde(rename = "it")]
    pub iteration: usize,

    #[serde(rename = "m")]
    pub measures: Vec<Measure>,
}

impl DataPoint {
    pub fn new(invocation: usize, iteration: usize) -> Self {
        Self { invocation, iteration, measures: Vec::new() }
    }

    pub fn add_point(&mut self, measure: Measure) {
        self.measures.push(measure);
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileElement {
    /// TODO: Percent of what?
    #[serde(rename = "p")]
    pub percent: usize,

    /// Method, Function, Symbol name
    #[serde(rename = "m")]
    pub symbol: String,

    /// Stack Trace
    #[serde(rename = "t")]
    pub stack_trace: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileData {
    /// Data.
    #[serde(rename = "d")]
    pub data: Vec<ProfileElement>,

    /// Invocation
    #[serde(rename = "in")]
    pub invocation: usize,

    /// Number of iterations
    #[serde(rename = "nit")]
    pub iterations: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    pub repo_url: Option<String>,
    pub branch_or_tag: String,
    pub commit_id: String,
    pub commit_msg: String,
    pub author_name: String,
    pub author_email: String,
    pub committer_name: String,
    pub committer_email: String,
}

impl Source {
    pub fn from_repo(
        path: impl AsRef<std::path::Path>,
    ) -> Result<(Self, OffsetDateTime), git2::Error> {
        use git2::Repository;

        let repo = Repository::open(path)?;
        let remote = repo.remotes()?;
        let remote = remote.get(0).expect("No remote associated to the repo");
        let remote = repo.find_remote(remote)?;

        let head = repo.head()?;
        let commit = head.peel_to_commit()?;

        let time = OffsetDateTime::from_unix_timestamp(commit.time().seconds()).unwrap();

        let author = commit.author();
        let committer = commit.committer();

        Ok((
            Self {
                repo_url: remote.url().map(|s| s.to_string()),
                branch_or_tag: head.name().unwrap().to_string(),
                commit_id: commit.id().to_string(),
                commit_msg: String::from_utf8_lossy(commit.message_bytes())
                    .to_string()
                    .lines()
                    .next()
                    .map_or(String::new(), |s| s.to_string()),
                author_name: author.name().unwrap().to_string(),
                author_email: author.email().unwrap().to_string(),
                committer_name: committer.name().unwrap().to_string(),
                committer_email: committer.email().unwrap().to_string(),
            },
            time,
        ))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    pub hostname: Option<String>,
    pub cpu: String,

    /// Advertised or nominal clock speed in Hertz.
    pub clock_speed: u64,

    /// Total number of bytes of memory provided by the system. */
    pub memory: u64,
    pub os_type: String,
    pub software: Vec<VersionInfo>,

    pub user_name: String,

    /// Is set true when the data was gathered by a manual run,
    /// possibly on a developer machine, instead of the usual benchmark server.
    pub manual_run: bool,
}

impl Environment {
    pub fn generate_from_current_config() -> Self {
        use sysinfo::System;

        let unknown_string = String::from("Unknown");
        let mut system = System::new();
        system.refresh_cpu();
        system.refresh_cpu_frequency();
        system.refresh_memory();

        let (cpu, frequency) = match system.cpus().first() {
            Some(cpu) => (
                format!("{} @ {:.2}GHz", cpu.brand(), cpu.frequency() as f64 / 1000.0),
                cpu.frequency() * 1_000_000,
            ),
            None => (unknown_string.clone(), 0),
        };

        let mut software = Vec::new();
        if let Some(distribution) = System::name() {
            software
                .push(VersionInfo { name: distribution, version: String::from("distribution") });
        }
        if let Some(kernel) = System::kernel_version() {
            software.push(VersionInfo { name: kernel, version: String::from("kernel") });
        }
        if let Some(os) = System::os_version() {
            software.push(VersionInfo { name: os, version: String::from("kernel-release") });
        }
        if let Some(arch) = System::cpu_arch() {
            software.push(VersionInfo { name: arch, version: String::from("arch") });
        }

        Self {
            hostname: System::host_name(),
            cpu,
            clock_speed: frequency,
            memory: system.total_memory(),
            os_type: System::long_os_version().unwrap_or(unknown_string.clone()),
            user_name: System::name().unwrap_or(unknown_string.clone()),
            manual_run: false,
            software,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Criterion {
    /// Id used to identify a criterion tuple.
    #[serde(rename = "i")]
    pub id: usize,
    /// Name of the criterion
    #[serde(rename = "c")]
    pub name: String,
    /// Unit of the criterion
    #[serde(rename = "u")]
    pub unit: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Unit {
    pub name: String,
    pub description: String,
    pub less_is_better: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkId {
    /// benchmark name
    #[serde(rename = "b")]
    pub benchmark_name: String,

    /// exe name
    #[serde(rename = "e")]
    pub exe_name: String,

    /// suite name
    #[serde(rename = "s")]
    pub suite_name: String,

    /// varValue
    #[serde(rename = "v")]
    pub var_value: Option<String>,

    /// cores
    #[serde(rename = "c")]
    pub cores: Option<String>,

    /// input size
    #[serde(rename = "i")]
    pub input_size: Option<String>,

    /// extra args
    #[serde(rename = "ea")]
    pub extra_argsa: Option<String>,
}
