//! Runs real monty workers, so these tests need a binary: `CODEX_MONTY_BINARY` or `monty`
//! on `PATH`. Without one they pass trivially.
//!
//! The pool is process-wide and bound to the tokio runtime that started it, and every
//! `#[tokio::test]` gets its own runtime, so the scenarios run in sequence inside one test.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use pretty_assertions::assert_eq;
use tempfile::TempDir;

use super::MontyRequest;
use super::MontyRunOutput;
use super::run_python;
use super::virtual_cwd;

fn binary() -> Option<PathBuf> {
    std::env::var(super::MONTY_BINARY_ENV)
        .ok()
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .or_else(|| which::which("monty").ok())
}

async fn run(binary: &Path, code: &str, cwd: &Path, writable: bool) -> MontyRunOutput {
    run_python(
        binary.to_path_buf(),
        MontyRequest {
            code,
            cwd,
            writable,
            timeout: Duration::from_secs(10),
        },
    )
    .await
    .expect("monty should run")
}

#[tokio::test(flavor = "multi_thread")]
async fn runs_python_in_monty_workers() {
    let Some(binary) = binary() else {
        return;
    };
    let dir = TempDir::new().unwrap();

    // Prints and the value of the final expression.
    let output = run(&binary, "print('hello')\nx = 20\nx + 1", dir.path(), false).await;
    assert_eq!(output.output, "hello\n");
    assert_eq!(output.result.as_deref(), Some("21"));
    assert!(output.exception.is_none());
    assert_eq!(output.exit_code(), 0);

    // Reads and writes through the mounted directory, which is also the cwd.
    std::fs::write(dir.path().join("in.txt"), "data").unwrap();
    let code = "from pathlib import Path\nimport os\nprint(os.getcwd())\nPath('out.txt').write_text(Path('in.txt').read_text().upper())\n";
    let output = run(&binary, code, dir.path(), true).await;
    assert!(output.exception.is_none(), "{}", output.render());
    assert_eq!(output.output.trim(), virtual_cwd(dir.path()));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
        "DATA"
    );

    // A read-only mount rejects writes.
    let code = "from pathlib import Path\nPath('denied.txt').write_text('x')\n";
    let output = run(&binary, code, dir.path(), false).await;
    assert_eq!(output.exit_code(), 1);
    assert!(
        output.render().contains("PermissionError"),
        "{}",
        output.render()
    );
    assert!(!dir.path().join("denied.txt").exists());

    // Exceptions render as tracebacks after the output that preceded them.
    let output = run(
        &binary,
        "print('before')\n{}['missing']\n",
        dir.path(),
        false,
    )
    .await;
    assert_eq!(output.exit_code(), 1);
    assert!(!output.likely_unsupported());
    let rendered = output.render();
    assert!(rendered.starts_with("before\nTraceback"), "{rendered}");
    assert!(rendered.contains("KeyError"), "{rendered}");

    // Missing modules and paths outside the mount look like gaps in Monty.
    let output = run(&binary, "import requests\n", dir.path(), false).await;
    assert!(output.likely_unsupported(), "{}", output.render());
    let output = run(&binary, "open('/etc/hosts').read()\n", dir.path(), false).await;
    assert!(output.likely_unsupported(), "{}", output.render());
    assert!(
        output.render().contains("PermissionError"),
        "{}",
        output.render()
    );

    // sys.exit sets the status.
    let output = run(
        &binary,
        "print('x')\nraise SystemExit('3')\n",
        dir.path(),
        false,
    )
    .await;
    assert_eq!(output.exit_code(), 3, "{:?}", output.exception);
    assert_eq!(output.render(), "x\n");
}

#[test]
fn virtual_paths() {
    assert_eq!(virtual_cwd(Path::new("/a/b/")), "/a/b");
    assert_eq!(virtual_cwd(Path::new("C:\\a")), "/workspace");
}
