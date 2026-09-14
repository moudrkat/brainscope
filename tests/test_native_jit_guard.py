"""The torch>=2.14 Triton guard decides before `import torch` and must only
act when the C toolchain for Triton's driver shim is missing."""
import importlib
import os
import sys


def _fresh_guard(monkeypatch, header_exists: bool, compiler: str | None):
    monkeypatch.delenv("TORCH_DISABLE_NATIVE_JIT", raising=False)
    import brainscope.server as srv
    monkeypatch.setattr(os.path, "exists", lambda p: header_exists if p.endswith("Python.h") else True)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: compiler)
    return srv._native_jit_guard()


def test_header_missing_disables(monkeypatch):
    reason = _fresh_guard(monkeypatch, header_exists=False, compiler="/usr/bin/gcc")
    assert reason and "Python.h" in reason
    assert os.environ["TORCH_DISABLE_NATIVE_JIT"] == "1"


def test_no_compiler_disables(monkeypatch):
    reason = _fresh_guard(monkeypatch, header_exists=True, compiler=None)
    assert reason and "compiler" in reason
    assert os.environ["TORCH_DISABLE_NATIVE_JIT"] == "1"


def test_toolchain_present_leaves_torch_alone(monkeypatch):
    reason = _fresh_guard(monkeypatch, header_exists=True, compiler="/usr/bin/gcc")
    assert reason is None
    assert "TORCH_DISABLE_NATIVE_JIT" not in os.environ


def test_user_choice_wins(monkeypatch):
    monkeypatch.setenv("TORCH_DISABLE_NATIVE_JIT", "0")
    import brainscope.server as srv
    assert srv._native_jit_guard() is None
    assert os.environ["TORCH_DISABLE_NATIVE_JIT"] == "0"
