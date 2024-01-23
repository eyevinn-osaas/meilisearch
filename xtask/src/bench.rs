use std::path::PathBuf;

use clap::Parser;

pub fn default_http_addr() -> String {
    "127.0.0.1:7700".to_string()
}
pub fn default_report_folder() -> String {
    "./report/".into()
}

/// Run benchmarks from a workload
#[derive(Parser, Debug)]
pub struct BenchDeriveArgs {
    /// Filename of the workload file, pass multiple filenames
    /// to run multiple workloads in the specified order.
    ///
    /// Each workload will get its own report file.
    #[arg(value_name = "WORKLOAD_FILE", last = true)]
    workload_file: Vec<PathBuf>,

    /// Directory where the reports will be output.
    #[arg(short, long, default_value_t = default_report_folder())]
    report_folder: String,
}

pub struct Workload {
    pub name: String,
    pub run_count: u16,
    pub extra_cli_args: Vec<String>,
    pub commands: Vec<Command>,
}

pub struct Command {
    pub route: String,
    pub method: Method,
    pub body: serde_json::Value,
    pub synchronous: SyncMode,
}

pub enum Method {
    GET,
    POST,
    PATCH,
    DELETE,
}

pub enum SyncMode {
    DontWait,
    WaitForResponse,
    WaitForTask,
}

pub fn run(args: BenchDeriveArgs) -> anyhow::Result<()> {
    // delete DB
    // start Meilisearch with extra args
    // loop on health until it's ready

    // execute commands
    Ok(())
}
