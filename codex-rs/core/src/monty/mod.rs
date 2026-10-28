//! Python execution through a pool of Monty sandbox workers.
//!
//! [Monty](https://github.com/pydantic/monty) is a Python interpreter written in Rust.
//! `monty-pool` runs it in worker subprocesses that are prewarmed, so a snippet starts in
//! well under a millisecond instead of paying CPython's startup cost. The sandbox reaches
//! the host only through the mounts it is given: here the working directory, mounted at
//! its own path. There is no network, no environment and no way to spawn processes.
//!
//! Two callers share the pool: the `run_python` tool and the `exec_command` handler, which
//! diverts shell commands that only run an inline Python script (see [`python_command`]).

pub(crate) mod python_command;
#[cfg(test)]
mod tests;

use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use monty_pool::MountSpec;
use monty_pool::MountSpecMode;
use monty_pool::Pool;
use monty_pool::PoolConfig;
use monty_pool::PoolError;
use monty_pool::ReplConfig;
use monty_pool::ResumeValue;
use monty_pool::TurnEvent;
use monty_pool::on_print_sync;
use monty_types::ExcType;
use monty_types::MontyException;
use monty_types::MontyObject;
use monty_types::NamedValues;
use monty_types::ResourceLimits;
use tokio::sync::OnceCell;

use crate::config::Config;

/// Names the worker binary, taking precedence over `monty_binary` in `config.toml`.
pub(crate) const MONTY_BINARY_ENV: &str = "CODEX_MONTY_BINARY";
/// Execution-time budget for one snippet when the caller does not set one. Time spent
/// in host filesystem calls does not count.
pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
/// Longest budget a caller may ask for.
pub(crate) const MAX_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// Live allocations a worker may hold; the worker exits past it.
const MAX_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
/// Every filesystem call is a suspension. A script walking a repository makes far more
/// than monty's default budget of 1000.
const MAX_SUSPENSIONS: usize = 1_000_000;
/// Workers kept warm beyond the one `PoolConfig::subprocess` starts.
const MIN_PROCESSES: usize = 2;
const CHECKOUT_TIMEOUT: Duration = Duration::from_secs(30);
/// Headroom between the in-sandbox budget and the pool killing the worker.
const REQUEST_TIMEOUT_GRACE: Duration = Duration::from_secs(15);
/// Name the sandbox reports in tracebacks.
const SCRIPT_NAME: &str = "<stdin>";

/// One snippet to run.
pub(crate) struct MontyRequest<'a> {
    pub code: &'a str,
    /// Host directory mounted into the sandbox and used as its working directory.
    pub cwd: &'a Path,
    /// Whether the sandbox may write under `cwd`.
    pub writable: bool,
    /// Execution-time budget; clamped to [`MAX_TIMEOUT`].
    pub timeout: Duration,
}

/// What a snippet produced.
#[derive(Debug)]
pub(crate) struct MontyRunOutput {
    /// `print()` output, stdout and stderr interleaved in order.
    pub output: String,
    /// `repr()` of the value of the final expression, when it was not `None`.
    pub result: Option<String>,
    /// The uncaught exception, if the snippet raised.
    pub exception: Option<MontyException>,
    pub duration: Duration,
}

impl MontyRunOutput {
    /// The exit status `python` would have reported.
    pub fn exit_code(&self) -> i32 {
        match &self.exception {
            None => 0,
            Some(exc) if exc.exc_type() == ExcType::SystemExit => {
                system_exit_code(exc.message()).unwrap_or(1)
            }
            Some(_) => 1,
        }
    }

    /// Output as a terminal would show it: the prints, then the traceback.
    pub fn render(&self) -> String {
        let mut text = self.output.clone();
        if let Some(exc) = &self.exception {
            if exc.exc_type() == ExcType::SystemExit {
                // `sys.exit("message")` prints the message; `sys.exit(2)` prints nothing.
                if let (Some(message), None) = (exc.message(), system_exit_code(exc.message())) {
                    push_line(&mut text, message);
                }
            } else {
                push_line(&mut text, &exc.to_string());
            }
        }
        text
    }

    /// Whether the failure looks like a gap in Monty rather than a bug in the script, so the
    /// same code may well run under CPython. Used to decide whether a diverted shell command
    /// should fall back to the real interpreter.
    pub fn likely_unsupported(&self) -> bool {
        self.exception.as_ref().is_some_and(|exc| {
            matches!(
                exc.exc_type(),
                ExcType::NotImplementedError
                    | ExcType::ImportError
                    | ExcType::ModuleNotFoundError
                    | ExcType::NameError
                    | ExcType::AttributeError
                    | ExcType::PermissionError
                    | ExcType::SyntaxError
                    | ExcType::TimeoutError
                    | ExcType::MemoryError
            )
        })
    }
}

/// Reads the status out of a `SystemExit` message: `None` and `0` exit cleanly, an integer
/// is the status, anything else is a message printed before exiting with 1.
fn system_exit_code(message: Option<&str>) -> Option<i32> {
    match message {
        None | Some("None") => Some(0),
        Some(text) => text.trim().parse().ok(),
    }
}

fn push_line(text: &mut String, line: &str) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(line);
    if !text.ends_with('\n') {
        text.push('\n');
    }
}

/// Monty could not run the snippet at all: no worker binary, or the pool failed. The
/// snippet itself has not been judged.
#[derive(Debug)]
pub(crate) struct MontyUnavailable(pub String);

impl fmt::Display for MontyUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<PoolError> for MontyUnavailable {
    fn from(err: PoolError) -> Self {
        Self(err.to_string())
    }
}

/// Finds the worker binary: the environment variable, then `monty_binary` from config, then
/// `monty` on `PATH`.
pub(crate) fn resolve_binary(config: &Config) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(MONTY_BINARY_ENV)
        && !path.trim().is_empty()
    {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = &config.monty_binary {
        return Some(path.to_path_buf());
    }
    which::which("monty").ok()
}

/// Sandbox-side path of the mounted working directory. Host paths that are already POSIX
/// keep their spelling, so absolute paths the model reads out of the repository work
/// unchanged; anything else is mounted at `/workspace`.
pub(crate) fn virtual_cwd(cwd: &Path) -> String {
    let text = cwd.to_string_lossy();
    if text.starts_with('/') && !text.contains('\0') {
        text.trim_end_matches('/').to_owned()
    } else {
        "/workspace".to_owned()
    }
}

static POOL: OnceCell<Arc<Pool>> = OnceCell::const_new();

/// The process-wide pool, started on first use with the binary that caller named. A failed
/// start is not cached, so a later call retries.
async fn shared_pool(binary: PathBuf) -> Result<Arc<Pool>, MontyUnavailable> {
    if tokio::runtime::Handle::current().runtime_flavor()
        == tokio::runtime::RuntimeFlavor::CurrentThread
    {
        return Err(MontyUnavailable(
            "monty workers need a multi-threaded tokio runtime".to_owned(),
        ));
    }
    POOL.get_or_try_init(|| async move {
        if !binary.is_file() {
            return Err(MontyUnavailable(format!(
                "monty binary `{}` does not exist",
                binary.display()
            )));
        }
        let mut config = PoolConfig::subprocess(&binary);
        config.min_processes = MIN_PROCESSES;
        config.max_processes = config.max_processes.max(MIN_PROCESSES);
        config.checkout_timeout = Some(CHECKOUT_TIMEOUT);
        config.request_timeout = Some(MAX_TIMEOUT + REQUEST_TIMEOUT_GRACE);
        let pool = Pool::new(config).await.map_err(|err| {
            MontyUnavailable(format!(
                "failed to start monty workers from `{}`: {err}",
                binary.display()
            ))
        })?;
        tracing::info!(binary = %binary.display(), "started monty worker pool");
        Ok(Arc::new(pool))
    })
    .await
    .cloned()
}

/// Runs one snippet on a pooled worker and returns the worker when done.
pub(crate) async fn run_python(
    binary: PathBuf,
    request: MontyRequest<'_>,
) -> Result<MontyRunOutput, MontyUnavailable> {
    let pool = shared_pool(binary).await?;
    let cwd = request.cwd.to_path_buf();
    let virtual_cwd = virtual_cwd(&cwd);
    let mode = if request.writable {
        MountSpecMode::ReadWrite
    } else {
        MountSpecMode::ReadOnly
    };
    // Opening the directory is blocking I/O.
    let mount_cwd = virtual_cwd.clone();
    let mount = tokio::task::spawn_blocking(move || MountSpec::new(&mount_cwd, &cwd, mode))
        .await
        .map_err(|err| MontyUnavailable(format!("mount task failed: {err}")))?
        .map_err(|err| MontyUnavailable(format!("cannot mount working directory: {err}")))?;

    let limits = ResourceLimits {
        max_feed_duration: Some(request.timeout.min(MAX_TIMEOUT)),
        max_memory: Some(MAX_MEMORY_BYTES),
        max_suspensions: MAX_SUSPENSIONS,
        ..ResourceLimits::default()
    };
    let repl = ReplConfig {
        script_name: SCRIPT_NAME.to_owned(),
        limits: Some(limits),
        ..ReplConfig::default()
    };
    let mut checkout = pool.checkout(&repl).await?;

    let output = Arc::new(Mutex::new(String::new()));
    let sink = Arc::clone(&output);
    let mut on_print = on_print_sync(move |_stream, text| {
        if let Ok(mut buffer) = sink.lock() {
            buffer.push_str(text);
        }
    });

    let started = Instant::now();
    let mut event = checkout
        .feed_with_cwd(
            request.code,
            NamedValues::default(),
            vec![mount],
            Some(&virtual_cwd),
            /*skip_type_check*/ true,
            &mut on_print,
        )
        .await;
    let finished = loop {
        match event {
            Ok(TurnEvent::Complete(value)) => break Ok(Finished::Value(value)),
            Ok(TurnEvent::OsCall { system_sleep, .. }) => {
                if let Some(delay) = system_sleep {
                    tokio::time::sleep(delay).await;
                    event = checkout
                        .resume(ResumeValue::Return(MontyObject::none()), &mut on_print)
                        .await;
                    continue;
                }
                event = match checkout.resume_from_mounts(&mut on_print).await {
                    Ok(Some(next)) => Ok(next),
                    // Outside the mount: the sandbox raises its own PermissionError.
                    Ok(None) => {
                        checkout
                            .resume(ResumeValue::NotHandled, &mut on_print)
                            .await
                    }
                    Err(err) => Err(err),
                };
            }
            // No host functions are exposed; the sandbox raises NameError.
            Ok(TurnEvent::FunctionCall { .. }) => {
                event = checkout.resume(ResumeValue::NotFound, &mut on_print).await;
            }
            Ok(TurnEvent::NameLookup { .. }) => {
                event = checkout
                    .resume_name_lookup(None::<MontyObject>, &mut on_print)
                    .await;
            }
            Ok(TurnEvent::ResolveFutures { .. }) => {
                break Err(PoolError::Protocol(
                    "the sandbox awaited a host future, which codex does not provide".into(),
                ));
            }
            Err(PoolError::Runtime(exc)) => break Ok(Finished::Raised(exc)),
            Err(err) => break Err(err),
        }
    };
    let duration = started.elapsed();
    drop(on_print);

    if let Err(err) = checkout.finish().await {
        // The worker is discarded and replaced; the run's verdict stands.
        tracing::debug!("monty checkout did not return cleanly: {err}");
    }
    let finished = finished?;
    let output = output
        .lock()
        .map(|buffer| buffer.clone())
        .unwrap_or_default();
    let (result, exception) = match finished {
        Finished::Value(value) => (
            (value != MontyObject::none()).then(|| value.py_repr()),
            None,
        ),
        Finished::Raised(exc) => (None, Some(exc)),
    };
    Ok(MontyRunOutput {
        output,
        result,
        exception,
        duration,
    })
}

enum Finished {
    Value(MontyObject),
    Raised(MontyException),
}
