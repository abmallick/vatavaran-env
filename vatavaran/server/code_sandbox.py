"""Stateful IPython sandbox used by the Vatavaran environment."""

from __future__ import annotations

import contextlib
import io
import os
import re
import signal
import threading
import tempfile
import traceback
from pathlib import Path
from typing import Any

from IPython.terminal.embed import InteractiveShellEmbed

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback when tokenizer missing
    tiktoken = None


class _ExecutionTimeout(Exception):
    """Raised when code execution exceeds configured timeout."""


def _format_ipython_failure(execution: Any) -> str:
    """Best-effort traceback text from IPython ExecutionResult.

    IPython sets either ``error_in_exec`` (runtime) or ``error_before_exec``
    (syntax/transform). ``success`` is false if either is set; only reading
    ``error_in_exec`` can crash when the failure was ``error_before_exec``.
    """

    exc = execution.error_in_exec or execution.error_before_exec
    if exc is not None:
        tb = getattr(exc, "__traceback__", None)
        return "".join(
            traceback.format_exception(type(exc), exc, tb)
        ).strip()

    parts: list[str] = []
    info = getattr(execution, "info", None)
    if info is not None:
        parts.append(f"(IPython did not attach an exception; info={info!r})")
    else:
        parts.append("(IPython reported failure but no exception was attached.)")
    parts.append(f"ExecutionResult: {execution!r}")
    return "\n".join(parts)


class CodeSandbox:
    """Executes agent-provided Python code inside a persistent IPython shell."""

    def __init__(self, data_root: Path, sandbox_config: dict):
        self.data_root = Path(data_root).resolve()
        self.ipython_dir = (Path(tempfile.gettempdir()) / "vatavaran_ipython").resolve()
        self.ipython_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_config = sandbox_config
        self.output_token_limit = int(sandbox_config.get("output_token_limit", 16384))
        self.execution_timeout_sec = int(sandbox_config.get("execution_timeout_sec", 30))
        self.blocked_imports = sandbox_config.get("blocked_imports", [])
        self.forbid_write_patterns = [
            re.compile(pattern) for pattern in sandbox_config.get("forbid_write_patterns", [])
        ]
        self.pre_init_code = sandbox_config.get("pre_init_code", "import pandas as pd")
        self._tokenizer = self._init_tokenizer()
        self._kernel: InteractiveShellEmbed | None = None
        self._working_dir: Path = self.data_root

    @staticmethod
    def _init_tokenizer():
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Offline/sandbox fallback: use approximate token counting.
                return None

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is None:
            # Conservative fallback approximation.
            return max(1, len(text) // 4)
        return len(self._tokenizer.encode(text))

    def _timeout_handler(self, *_):
        raise _ExecutionTimeout("Code execution timed out.")

    @contextlib.contextmanager
    def _timeout(self):
        """SIGALRM-based timeout on Unix systems."""

        if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
            yield
            return
        previous = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.execution_timeout_sec)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    def _validate_code(self, code: str) -> str | None:
        for blocked in self.blocked_imports:
            if re.search(rf"\bimport\s+{re.escape(blocked)}\b", code):
                return f"Import blocked by policy: {blocked}"
            if re.search(rf"\bfrom\s+{re.escape(blocked)}\b", code):
                return f"Import blocked by policy: {blocked}"
        for pattern in self.forbid_write_patterns:
            if pattern.search(code):
                return (
                    "File write operation blocked by policy. "
                    "Use in-memory variables only."
                )
        return None

    def reset(self, working_dir: str | Path | None = None):
        """Reset kernel state and optional working directory."""

        if working_dir is not None:
            candidate = Path(working_dir).resolve()
            if not str(candidate).startswith(str(self.data_root)):
                candidate = self.data_root
            self._working_dir = candidate
        else:
            self._working_dir = self.data_root

        os.environ["IPYTHONDIR"] = str(self.ipython_dir)
        self._kernel = InteractiveShellEmbed()
        boot_code = (
            self.pre_init_code
            + f'\nimport os\nDATA_ROOT = r"{self.data_root}"\n'
            + f'os.chdir(r"{self._working_dir}")\n'
        )
        self._kernel.run_cell(boot_code)

    def execute(self, code: str) -> tuple[bool, str]:
        """Execute Python code in the persistent kernel."""

        if self._kernel is None:
            self.reset()

        error = self._validate_code(code)
        if error:
            return False, error

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                with self._timeout():
                    execution = self._kernel.run_cell(code)
        except _ExecutionTimeout as exc:
            return False, str(exc)
        except Exception:
            return False, traceback.format_exc()

        std_out = stdout_buffer.getvalue().strip()
        std_err = stderr_buffer.getvalue().strip()

        if not execution.success:
            error_text = _format_ipython_failure(execution)
            merged = "\n".join([part for part in [std_out, std_err, error_text] if part])
            return False, merged or "Execution failed."

        result_value = execution.result
        if result_value is None:
            merged = "\n".join([part for part in [std_out, std_err] if part])
            final_output = merged if merged else "Execution completed successfully."
        else:
            result_text = str(result_value).strip()
            merged = "\n".join([part for part in [result_text, std_out, std_err] if part])
            final_output = merged if merged else "Execution completed successfully."

        if self._count_tokens(final_output) > self.output_token_limit:
            return (
                False,
                f"Execution output exceeds token limit ({self.output_token_limit}).",
            )

        return True, final_output

    def list_files(self, relative_or_abs_path: str = "") -> tuple[bool, str]:
        """List directory contents under the allowed data root."""

        path = (relative_or_abs_path or "").strip()
        if not path or path == ".":
            target = self._working_dir
        else:
            candidate = Path(path)
            if not candidate.is_absolute():
                target = (self._working_dir / candidate).resolve()
            else:
                target = candidate.resolve()

        if not str(target).startswith(str(self.data_root)):
            return False, "Access denied: path must stay within telemetry data root."
        if not target.exists():
            return False, f"Path does not exist: {target}"
        if not target.is_dir():
            return False, f"Path is not a directory: {target}"

        rows = []
        for item in sorted(target.iterdir()):
            item_type = "dir" if item.is_dir() else "file"
            rows.append(f"{item_type}\t{item.name}")
        if not rows:
            return True, f"Directory is empty: {target}"
        return True, "\n".join(rows)
