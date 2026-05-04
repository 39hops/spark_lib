"""Bootstrap helpers for Azure Synapse notebooks.

Copy this file to ADLS, then load it from a notebook with:

    code = mssparkutils.fs.head("abfss://.../synapse_bootstrap.py", 200000)
    exec(code)

The helpers below are intentionally standalone so they work before spark_lib is
installed or importable.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any, Dict, Optional


def install_wheel_from_abfss(
    wheel_path: str,
    *,
    force_reinstall: bool = True,
    target_dir: str = "/tmp/spark_lib_wheels",
) -> None:
    """Copy a wheel from ADLS to the driver and install it with pip."""
    local_path = _copy_to_local(wheel_path, target_dir)
    cmd = [sys.executable, "-m", "pip", "install"]
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.append(local_path)
    subprocess.check_call(cmd)
    importlib.invalidate_caches()


def exec_py_from_abfss(
    py_path: str,
    namespace: Optional[Dict[str, Any]] = None,
    *,
    target_dir: str = "/tmp/spark_lib_sources",
) -> Dict[str, Any]:
    """Copy a Python file from ADLS and execute it in `namespace`."""
    local_path = _copy_to_local(py_path, target_dir)
    with open(local_path, "r", encoding="utf-8") as f:
        code = compile(f.read(), py_path, "exec")
    ns: Dict[str, Any] = namespace if namespace is not None else {}
    exec(code, ns)
    return ns


def _copy_to_local(path: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, os.path.basename(path.rstrip("/")))
    _nbutils().fs.cp(path, "file://" + local_path, recurse=False)
    return local_path


def _nbutils() -> Any:
    try:
        from notebookutils import mssparkutils
        return mssparkutils
    except ImportError:
        try:
            import mssparkutils  # type: ignore
            return mssparkutils
        except ImportError:
            raise RuntimeError("mssparkutils is required in Synapse notebooks")
