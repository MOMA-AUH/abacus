"""Run external tools with diagnostics visible at the default log level."""

import shlex
import subprocess

from abacus.logging import logger


def run_tool(command: list[str], *, input: str | None = None) -> subprocess.CompletedProcess[str]:
    """Capture tool output, logging failures without requiring verbose mode."""
    try:
        process = subprocess.run(command, input=input, text=True, capture_output=True, check=False)
    except OSError as error:
        message = f"Could not start {command[0]}: {error}"
        logger.error("%s\nCommand: %s", message, shlex.join(command))
        raise RuntimeError(message) from error

    if process.returncode != 0:
        message = f"{command[0]} failed with exit code {process.returncode}"
        logger.error("%s\nCommand: %s", message, shlex.join(command))
        if process.stderr:
            logger.error("%s stderr:\n%s", command[0], process.stderr)
        if process.stdout:
            logger.error("%s stdout:\n%s", command[0], process.stdout)
        raise RuntimeError(message)
    return process
