"""Supplied Git Bash runtime. No model wrapper, parser, or agent loop."""

import os
from pathlib import Path
import shutil
import subprocess


def find_bash():
    """Select Git Bash on Windows, never system32/bash.exe (the WSL launcher)."""
    override = os.environ.get("GIT_BASH")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"GIT_BASH does not point to a file: {path}")
        return str(path.resolve())
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/bin/bash.exe",
        ]
        git = shutil.which("git")
        if git:
            candidates.append(Path(git).parent.parent / "bin/bash.exe")
        for path in candidates:
            if path.is_file():
                return str(path.resolve())
        raise FileNotFoundError("Install Git for Windows or set GIT_BASH to its bin/bash.exe.")
    path = shutil.which("bash")
    if not path:
        raise FileNotFoundError("Bash is not installed.")
    return path


def confirm(command):
    print("\nCommand proposed for YOUR HOST:\n" + command)
    return input("Type yes to run it: ").strip().lower() == "yes"


class Bash:
    """bash(command) -> {stdout, stderr, exit_code[, error]}. NOT a sandbox.

    Approval does not alter the command. A fresh shell starts in workspace for
    every call. No process-tree isolation or streaming output limit is claimed.
    """

    def __init__(self, workspace, *, approve=confirm, timeout=20, output_limit=4000):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise NotADirectoryError(self.workspace)
        self.executable = find_bash()
        self.approve = approve
        self.timeout = timeout
        self.output_limit = output_limit

    def __call__(self, command):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash expects a nonempty command string")
        if not self.approve(command):
            return {"stdout": "", "stderr": "", "exit_code": None, "error": "User denied execution. Do not retry this command."}

        def clip(text):
            text = text or ""
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            return text[:self.output_limit] + ("\n[truncated]" if len(text) > self.output_limit else "")

        env = os.environ.copy()
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)
        if os.name == "nt":
            root = Path(self.executable).parent.parent
            env["PATH"] = os.pathsep.join([str(root / "usr/bin"), str(root / "bin"), env.get("PATH", "")])
        try:
            result = subprocess.run(
                [self.executable, "--noprofile", "--norc", "-c", command],
                cwd=self.workspace, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout, shell=False,
            )
            return {"stdout": clip(result.stdout), "stderr": clip(result.stderr), "exit_code": result.returncode}
        except subprocess.TimeoutExpired as exc:
            return {"stdout": clip(exc.stdout), "stderr": clip(exc.stderr), "exit_code": None, "error": "Command timed out"}
        except OSError as exc:
            return {"stdout": "", "stderr": "", "exit_code": None, "error": str(exc)}

