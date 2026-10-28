//! Recognizes shell commands that do nothing but run an inline Python script, so the
//! script can be run in Monty instead of a shell plus CPython.
//!
//! Accepted shapes, each as the whole script of a `bash -lc` / `zsh -lc` / `sh -c` command:
//!
//! ```text
//! python3 - <<'PY'        # heredoc on stdin, with or without the `-`
//! ...
//! PY
//! python3 -c 'print(1)'   # inline code
//! python3 script.py       # a file under the working directory
//! ```
//!
//! The interpreter may be `python`, `python3`, `python3.N`, a path to one of those, or any
//! of them behind `uv run`, with the flags that do not change what the script does.
//! Anything else (pipes, redirections, arguments, env assignments, unquoted heredocs that
//! expand shell variables) is left to the shell.

use std::path::Path;

use codex_shell_command::parse_command::extract_shell_command;

/// Where the Python came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PythonSource {
    /// The script text: a heredoc body or a `-c` argument.
    Inline(String),
    /// A `.py` path as written on the command line, relative to the command's cwd.
    File(String),
}

/// The Python a `shell -c script` command runs, when that is all it does.
pub(crate) fn python_source_from_command(command: &[String]) -> Option<PythonSource> {
    let (shell, script) = extract_shell_command(command)?;
    let shell_name = Path::new(shell)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(shell);
    if !matches!(shell_name, "bash" | "zsh" | "sh" | "dash") {
        return None;
    }
    python_source_from_script(script)
}

/// The Python a shell script runs, when that is all it does.
pub(crate) fn python_source_from_script(script: &str) -> Option<PythonSource> {
    let script = script.trim();
    if let Some(source) = heredoc_script(script) {
        return Some(source);
    }
    let words = shell_words(script)?;
    match strip_interpreter(&words)? {
        [flag, code] if flag == "-c" => Some(PythonSource::Inline(code.clone())),
        [path] if path.ends_with(".py") && !path.starts_with('-') => {
            Some(PythonSource::File(path.clone()))
        }
        _ => None,
    }
}

/// `python3 - <<'PY' ... PY`, the body ending the script.
fn heredoc_script(script: &str) -> Option<PythonSource> {
    let (first_line, body) = script.split_once('\n')?;
    let (left, right) = first_line.split_once("<<")?;
    let (strip_tabs, right) = match right.strip_prefix('-') {
        Some(right) => (true, right),
        None => (false, right),
    };
    let (delimiter, quoted) = heredoc_delimiter(right.trim())?;
    let words = shell_words(left)?;
    match strip_interpreter(&words)? {
        [] => {}
        [dash] if dash == "-" => {}
        _ => return None,
    }

    let mut lines = Vec::new();
    let mut remaining = body.lines();
    let mut terminated = false;
    for line in remaining.by_ref() {
        let line = if strip_tabs {
            line.trim_start_matches('\t')
        } else {
            line
        };
        if line == delimiter {
            terminated = true;
            break;
        }
        lines.push(line);
    }
    if !terminated || remaining.any(|line| !line.trim().is_empty()) {
        return None;
    }
    let mut code = lines.join("\n");
    code.push('\n');
    // An unquoted delimiter means the shell expands `$var`, `$(cmd)` and backquotes in
    // the body, and removes backslashes; Monty would see the raw text.
    if !quoted && code.contains(['$', '`', '\\']) {
        return None;
    }
    Some(PythonSource::Inline(code))
}

/// The delimiter word after `<<`, and whether it was quoted. Nothing may follow it.
fn heredoc_delimiter(word: &str) -> Option<(String, bool)> {
    if word.is_empty() || word.chars().any(char::is_whitespace) {
        return None;
    }
    for quote in ['\'', '"'] {
        if let Some(inner) = word
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return (!inner.is_empty() && !inner.contains(quote)).then(|| (inner.to_owned(), true));
        }
    }
    if let Some(inner) = word.strip_prefix('\\') {
        return is_plain_word(inner).then(|| (inner.to_owned(), true));
    }
    is_plain_word(word).then(|| (word.to_owned(), false))
}

fn is_plain_word(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The words after the interpreter and its harmless flags; `None` when the command does not
/// start with a bare Python interpreter.
fn strip_interpreter(words: &[String]) -> Option<&[String]> {
    let mut rest = words;
    if let [uv, run, tail @ ..] = rest
        && uv == "uv"
        && run == "run"
    {
        rest = tail;
    }
    let (interpreter, mut rest) = rest.split_first()?;
    if !is_python_interpreter(interpreter) {
        return None;
    }
    while let [flag, tail @ ..] = rest
        && matches!(
            flag.as_str(),
            "-u" | "-B" | "-I" | "-E" | "-s" | "-S" | "-q" | "-O" | "-OO"
        )
    {
        rest = tail;
    }
    Some(rest)
}

fn is_python_interpreter(word: &str) -> bool {
    let name = word.rsplit(['/', '\\']).next().unwrap_or(word);
    let name = name.strip_suffix(".exe").unwrap_or(name);
    let Some(version) = name.strip_prefix("python") else {
        return false;
    };
    version.is_empty()
        || version == "3"
        || version
            .strip_prefix("3.")
            .is_some_and(|minor| !minor.is_empty() && minor.bytes().all(|b| b.is_ascii_digit()))
}

/// Splits one plain command into words, honouring single quotes, double quotes and
/// backslashes. Returns `None` for anything the shell would treat specially: operators,
/// redirections, globs, expansions, comments, tildes and braces.
fn shell_words(input: &str) -> Option<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '\'' => break,
                        ch => current.push(ch),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '"' => break,
                        '\\' => {
                            let next = chars.next()?;
                            if !matches!(next, '"' | '\\' | '$' | '`' | '\n') {
                                current.push('\\');
                            }
                            if next != '\n' {
                                current.push(next);
                            }
                        }
                        '$' | '`' => return None,
                        ch => current.push(ch),
                    }
                }
            }
            '\\' => {
                in_word = true;
                let next = chars.next()?;
                if next != '\n' {
                    current.push(next);
                }
            }
            c if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut current));
                    in_word = false;
                }
            }
            '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' | '#' | '*' | '?' | '[' | '~'
            | '{' | '}' | '!' => return None,
            c => {
                in_word = true;
                current.push(c);
            }
        }
    }
    if in_word {
        words.push(current);
    }
    Some(words)
}

#[cfg(test)]
mod tests {
    use super::PythonSource;
    use super::python_source_from_command;
    use super::python_source_from_script;
    use pretty_assertions::assert_eq;

    fn inline(code: &str) -> Option<PythonSource> {
        Some(PythonSource::Inline(code.to_owned()))
    }

    #[test]
    fn quoted_heredoc_on_stdin() {
        let script = "python3 - <<'PY'\nfrom datetime import date\nprint(date(2027, 1, 1))\nPY\n";
        assert_eq!(
            python_source_from_script(script),
            inline("from datetime import date\nprint(date(2027, 1, 1))\n")
        );
        let command = vec!["bash".to_owned(), "-lc".to_owned(), script.to_owned()];
        assert_eq!(
            python_source_from_command(&command),
            inline("from datetime import date\nprint(date(2027, 1, 1))\n")
        );
    }

    #[test]
    fn heredoc_variants() {
        assert_eq!(
            python_source_from_script("python <<\"EOF\"\nprint('x')\nEOF"),
            inline("print('x')\n")
        );
        assert_eq!(
            python_source_from_script("python3.14 - <<-PY\n\tprint('x')\n\tPY"),
            inline("print('x')\n")
        );
        assert_eq!(
            python_source_from_script("uv run python -u - <<\\PY\nprint('x')\nPY"),
            inline("print('x')\n")
        );
        assert_eq!(
            python_source_from_script("/usr/bin/python3 -<<'PY'\nprint('x')\nPY"),
            inline("print('x')\n")
        );
        // An unquoted delimiter is fine while the body has nothing the shell would expand.
        assert_eq!(
            python_source_from_script("python3 - <<PY\nprint(1 + 1)\nPY"),
            inline("print(1 + 1)\n")
        );
    }

    #[test]
    fn heredoc_rejections() {
        // The shell would expand `$HOME`.
        assert_eq!(
            python_source_from_script("python3 - <<PY\nprint('$HOME')\nPY"),
            None
        );
        // Unterminated.
        assert_eq!(
            python_source_from_script("python3 - <<'PY'\nprint(1)\n"),
            None
        );
        // Another command follows the heredoc.
        assert_eq!(
            python_source_from_script("python3 - <<'PY'\nprint(1)\nPY\nls"),
            None
        );
        // Output is redirected or piped.
        assert_eq!(
            python_source_from_script("python3 - <<'PY' | head\nprint(1)\nPY"),
            None
        );
        assert_eq!(
            python_source_from_script("python3 - 2>&1 <<'PY'\nprint(1)\nPY"),
            None
        );
        // Script arguments.
        assert_eq!(
            python_source_from_script("python3 - arg <<'PY'\nprint(1)\nPY"),
            None
        );
        // Not python.
        assert_eq!(
            python_source_from_script("node - <<'JS'\nconsole.log(1)\nJS"),
            None
        );
        assert_eq!(
            python_source_from_script("cd x && python3 - <<'PY'\nprint(1)\nPY"),
            None
        );
    }

    #[test]
    fn dash_c_forms() {
        assert_eq!(
            python_source_from_script("python3 -c 'print(1)'"),
            inline("print(1)")
        );
        assert_eq!(
            python_source_from_script("python3 -c \"print('a b')\""),
            inline("print('a b')")
        );
        assert_eq!(
            python_source_from_script("python -c 'import json\nprint(json.dumps({}))'"),
            inline("import json\nprint(json.dumps({}))")
        );
        assert_eq!(
            python_source_from_script("python3 -c 'print(1<<2)'"),
            inline("print(1<<2)")
        );
        assert_eq!(
            python_source_from_script("python3 -c 'print(\"it'\\''s\")'"),
            inline("print(\"it's\")")
        );
        // Double quotes with an expansion.
        assert_eq!(
            python_source_from_script("python3 -c \"print('$X')\""),
            None
        );
        // Extra arguments or a different flag.
        assert_eq!(python_source_from_script("python3 -c 'print(1)' arg"), None);
        assert_eq!(python_source_from_script("python3 -m json.tool"), None);
        assert_eq!(
            python_source_from_script("python3 -c 'print(1)' > out"),
            None
        );
        assert_eq!(
            python_source_from_script("PYTHONPATH=. python3 -c 'print(1)'"),
            None
        );
    }

    #[test]
    fn file_form() {
        assert_eq!(
            python_source_from_script("python3 scripts/check.py"),
            Some(PythonSource::File("scripts/check.py".to_owned()))
        );
        assert_eq!(
            python_source_from_script("python3 scripts/check.py --flag"),
            None
        );
        assert_eq!(python_source_from_script("python3 ~/check.py"), None);
        assert_eq!(python_source_from_script("python3 -m pytest"), None);
        assert_eq!(python_source_from_script("python3"), None);
    }

    #[test]
    fn only_shell_wrappers_are_recognized() {
        let command = vec!["python3".to_owned(), "-c".to_owned(), "print(1)".to_owned()];
        assert_eq!(python_source_from_command(&command), None);
        let command = vec![
            "pwsh".to_owned(),
            "-Command".to_owned(),
            "python3 -c 'print(1)'".to_owned(),
        ];
        assert_eq!(python_source_from_command(&command), None);
    }
}
