//! A Code Mode session whose cells run Python on Monty workers.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use codex_code_mode_protocol::CellId;
use codex_code_mode_protocol::CodeModeSession;
use codex_code_mode_protocol::CodeModeSessionCellExecutionLimits;
use codex_code_mode_protocol::CodeModeSessionDelegate;
use codex_code_mode_protocol::CodeModeSessionResultFuture;
use codex_code_mode_protocol::DEFAULT_EXEC_YIELD_TIME_MS;
use codex_code_mode_protocol::ExecuteRequest;
use codex_code_mode_protocol::RuntimeResponse;
use codex_code_mode_protocol::StartedCell;
use codex_code_mode_protocol::TerminateRequest;
use codex_code_mode_protocol::WaitOutcome;
use codex_code_mode_protocol::WaitRequest;
use monty_pool::Pool;
use serde_json::Value as JsonValue;
use tokio_util::sync::CancellationToken;

use crate::cell::CellContext;
use crate::cell::CellOutcome;
use crate::cell::CellShared;
use crate::cell::run_cell;

type CellMap = Arc<Mutex<HashMap<CellId, Arc<CellHandle>>>>;

struct CellHandle {
    shared: Arc<CellShared>,
    cancel: CancellationToken,
}

/// When an observation returns before the cell finishes.
#[derive(Clone, Copy)]
enum Observe {
    /// After this long, after `yield_control()`, or when preempted.
    YieldAfter(Duration),
    /// Only once the cell has stopped.
    UntilStopped,
}

pub struct MontyCodeModeSession {
    pool: Arc<Pool>,
    limits: CodeModeSessionCellExecutionLimits,
    cells: CellMap,
    /// Values saved with `store()`, shared by every cell of the session.
    stored: Arc<Mutex<HashMap<String, JsonValue>>>,
    next_cell_id: AtomicU64,
    shutdown: CancellationToken,
}

impl MontyCodeModeSession {
    pub fn new(pool: Arc<Pool>, limits: CodeModeSessionCellExecutionLimits) -> Self {
        Self {
            pool,
            limits,
            cells: Arc::default(),
            stored: Arc::default(),
            next_cell_id: AtomicU64::new(1),
            shutdown: CancellationToken::new(),
        }
    }

    fn yield_after(&self, requested_ms: u64) -> Duration {
        let ms = match self.limits.max_yield_time_ms {
            Some(max) => requested_ms.min(max),
            None => requested_ms,
        };
        Duration::from_millis(ms)
    }

    fn cell(&self, cell_id: &CellId) -> Option<Arc<CellHandle>> {
        self.cells
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(cell_id)
            .cloned()
    }
}

impl CodeModeSession for MontyCodeModeSession {
    fn execute<'a>(
        &'a self,
        request: ExecuteRequest,
        delegate: Arc<dyn CodeModeSessionDelegate>,
        preempt: Option<CancellationToken>,
    ) -> CodeModeSessionResultFuture<'a, StartedCell> {
        Box::pin(async move {
            if self.shutdown.is_cancelled() {
                return Err("code-mode session is shut down".to_string());
            }
            let cell_id = CellId::new(
                self.next_cell_id
                    .fetch_add(1, Ordering::Relaxed)
                    .to_string(),
            );
            let handle = Arc::new(CellHandle {
                shared: Arc::default(),
                cancel: self.shutdown.child_token(),
            });
            self.cells
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(cell_id.clone(), Arc::clone(&handle));

            tokio::spawn(run_cell(CellContext {
                pool: Arc::clone(&self.pool),
                cell_id: cell_id.clone(),
                tool_call_id: request.tool_call_id,
                source: request.source,
                tools: request.enabled_tools,
                delegate,
                stored: Arc::clone(&self.stored),
                shared: Arc::clone(&handle.shared),
                cancel: handle.cancel.clone(),
                max_memory_bytes: self.limits.max_heap_size_bytes,
            }));

            let mode = Observe::YieldAfter(
                self.yield_after(request.yield_time_ms.unwrap_or(DEFAULT_EXEC_YIELD_TIME_MS)),
            );
            let cells = Arc::clone(&self.cells);
            let response_cell_id = cell_id.clone();
            Ok(StartedCell::from_future(cell_id, async move {
                Ok(observe(&cells, &response_cell_id, &handle, mode, preempt).await)
            }))
        })
    }

    fn wait<'a>(
        &'a self,
        request: WaitRequest,
        preempt: Option<CancellationToken>,
    ) -> CodeModeSessionResultFuture<'a, WaitOutcome> {
        Box::pin(async move {
            let Some(handle) = self.cell(&request.cell_id) else {
                return Ok(WaitOutcome::MissingCell(missing_cell(request.cell_id)));
            };
            let mode = Observe::YieldAfter(self.yield_after(request.yield_time_ms));
            Ok(WaitOutcome::LiveCell(
                observe(&self.cells, &request.cell_id, &handle, mode, preempt).await,
            ))
        })
    }

    fn terminate<'a>(
        &'a self,
        request: TerminateRequest,
    ) -> CodeModeSessionResultFuture<'a, WaitOutcome> {
        Box::pin(async move {
            let Some(handle) = self.cell(&request.cell_id) else {
                return Ok(WaitOutcome::MissingCell(missing_cell(request.cell_id)));
            };
            handle.cancel.cancel();
            Ok(WaitOutcome::LiveCell(
                observe(
                    &self.cells,
                    &request.cell_id,
                    &handle,
                    Observe::UntilStopped,
                    /*preempt*/ None,
                )
                .await,
            ))
        })
    }

    fn shutdown<'a>(&'a self) -> CodeModeSessionResultFuture<'a, ()> {
        Box::pin(async move {
            self.shutdown.cancel();
            self.cells
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clear();
            Ok(())
        })
    }
}

impl Drop for MontyCodeModeSession {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

/// Waits for the cell to finish or for a reason to yield, and returns its new output.
async fn observe(
    cells: &CellMap,
    cell_id: &CellId,
    handle: &CellHandle,
    mode: Observe,
    preempt: Option<CancellationToken>,
) -> RuntimeResponse {
    let preempt = preempt.unwrap_or_default();
    let deadline = match mode {
        Observe::YieldAfter(after) => Some(tokio::time::Instant::now() + after),
        Observe::UntilStopped => None,
    };
    loop {
        // Register for the next change before looking, so a change between the check
        // and the wait is not missed.
        let changed = handle.shared.changed.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        {
            let mut state = handle.shared.lock();
            if let Some(outcome) = state.outcome.take() {
                let content_items = state.drain_items();
                drop(state);
                cells
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(cell_id);
                return match outcome {
                    CellOutcome::Completed { error_text } => RuntimeResponse::Result {
                        cell_id: cell_id.clone(),
                        content_items,
                        error_text,
                        result_buffered_for: Duration::ZERO,
                        code_mode_host_duration: None,
                    },
                    CellOutcome::Terminated => RuntimeResponse::Terminated {
                        cell_id: cell_id.clone(),
                        content_items,
                        code_mode_host_duration: None,
                    },
                };
            }
            if matches!(mode, Observe::YieldAfter(_)) && state.yield_requested {
                state.yield_requested = false;
                return yielded(cell_id, state.drain_items());
            }
        }
        let timeout = async {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            () = &mut changed => {}
            () = timeout => return yielded(cell_id, handle.shared.lock().drain_items()),
            () = preempt.cancelled(), if deadline.is_some() => {
                return yielded(cell_id, handle.shared.lock().drain_items());
            }
        }
    }
}

fn yielded(
    cell_id: &CellId,
    content_items: Vec<codex_code_mode_protocol::FunctionCallOutputContentItem>,
) -> RuntimeResponse {
    RuntimeResponse::Yielded {
        cell_id: cell_id.clone(),
        content_items,
        code_mode_host_duration: None,
    }
}

fn missing_cell(cell_id: CellId) -> RuntimeResponse {
    let error_text = format!("exec cell {cell_id} not found");
    RuntimeResponse::Result {
        cell_id,
        content_items: Vec::new(),
        error_text: Some(error_text),
        result_buffered_for: Duration::ZERO,
        code_mode_host_duration: None,
    }
}
