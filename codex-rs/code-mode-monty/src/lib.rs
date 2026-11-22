//! Python Code Mode: `exec` cells run as Python scripts in
//! [Monty](https://github.com/pydantic/monty), a sandboxed Python interpreter written in Rust.
//!
//! Monty runs in pooled worker subprocesses (`monty-pool`), which are prewarmed, so a cell
//! starts in well under a millisecond. A worker reaches the host only through the
//! suspensions this crate answers: nested tool calls and helpers. It gets no mounts, so it
//! has no file system, network, environment or process access of its own.

mod cell;
mod convert;
mod session;

use std::path::PathBuf;
use std::sync::Arc;

use codex_code_mode_protocol::CodeModeSession;
use codex_code_mode_protocol::CodeModeSessionCellExecutionLimits;
use codex_code_mode_protocol::CodeModeSessionProvider;
use codex_code_mode_protocol::CodeModeSessionProviderFuture;
use monty_pool::Pool;
use monty_pool::PoolConfig;
use tokio::sync::OnceCell;

pub use session::MontyCodeModeSession;

/// Names the worker binary, taking precedence over configuration.
pub const MONTY_BINARY_ENV: &str = "CODEX_MONTY_BINARY";

/// Workers kept warm beyond the one `PoolConfig::subprocess` starts.
const MIN_PROCESSES: usize = 2;

/// Creates Monty-backed sessions. All sessions share one worker pool, started on first use.
pub struct MontyCodeModeSessionProvider {
    binary: Option<PathBuf>,
    pool: OnceCell<Arc<Pool>>,
}

impl MontyCodeModeSessionProvider {
    /// `binary` is the `monty` worker executable; see [`resolve_monty_binary`].
    pub fn new(binary: Option<PathBuf>) -> Self {
        Self {
            binary,
            pool: OnceCell::new(),
        }
    }

    async fn pool(&self) -> Result<Arc<Pool>, String> {
        self.availability()?;
        if tokio::runtime::Handle::current().runtime_flavor()
            == tokio::runtime::RuntimeFlavor::CurrentThread
        {
            return Err("Monty workers need a multi-threaded Tokio runtime".to_string());
        }
        let binary = self.binary.clone().unwrap_or_default();
        self.pool
            .get_or_try_init(|| async move {
                let mut config = PoolConfig::subprocess(&binary);
                config.min_processes = MIN_PROCESSES;
                config.max_processes = config.max_processes.max(MIN_PROCESSES);
                let pool = Pool::new(config).await.map_err(|err| {
                    format!(
                        "failed to start Monty workers from `{}`: {err}",
                        binary.display()
                    )
                })?;
                tracing::info!(binary = %binary.display(), "started Monty worker pool");
                Ok(Arc::new(pool))
            })
            .await
            .cloned()
    }
}

impl CodeModeSessionProvider for MontyCodeModeSessionProvider {
    fn availability(&self) -> Result<(), String> {
        match &self.binary {
            Some(binary) if binary.is_file() => Ok(()),
            Some(binary) => Err(format!(
                "Monty worker binary `{}` does not exist",
                binary.display()
            )),
            None => Err(format!(
                "no Monty worker binary found: set `{MONTY_BINARY_ENV}`, set \
                 `features.code_mode.monty_binary`, or put `monty` on PATH"
            )),
        }
    }

    fn create_session(&self) -> CodeModeSessionProviderFuture<'_> {
        self.create_session_with_limits(CodeModeSessionCellExecutionLimits::default())
    }

    fn create_session_with_limits<'a>(
        &'a self,
        limits: CodeModeSessionCellExecutionLimits,
    ) -> CodeModeSessionProviderFuture<'a> {
        Box::pin(async move {
            let pool = self.pool().await?;
            Ok(Arc::new(MontyCodeModeSession::new(pool, limits)) as Arc<dyn CodeModeSession>)
        })
    }
}

/// Finds the worker binary: [`MONTY_BINARY_ENV`], then `configured`, then `monty` on `PATH`.
pub fn resolve_monty_binary(configured: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(MONTY_BINARY_ENV).filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    if configured.is_some() {
        return configured;
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join("monty"))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(test)]
mod tests;
