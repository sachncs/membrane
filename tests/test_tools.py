"""The naming and docstring checkers that CI runs over membrane/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import check_docstrings
import check_naming


def write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(source)
    return path


def test_naming_flags_semi_private_names_in_nested_blocks(tmp_path: Path) -> None:
    source = '''"""Doc."""
try:
    _hidden = 1
except ImportError:
    _fallback = 2
if True:
    class _Klass: ...
__all__ = []
'''
    problems = "\n".join(check_naming.check_file(write(tmp_path, source)))
    assert "'_hidden'" in problems and "'_fallback'" in problems and "'_Klass'" in problems


def test_naming_allows_function_locals_and_mangled_names(tmp_path: Path) -> None:
    source = '''"""Doc."""
def f():
    _local = 1
    return _local
class C:
    def __init__(self):
        self.__private = 1
__all__ = ["C", "f"]
'''
    assert check_naming.check_file(write(tmp_path, source)) == []


def test_naming_flags_print_and_missing_all(tmp_path: Path) -> None:
    problems = "\n".join(check_naming.check_file(write(tmp_path, '"""Doc."""\nprint("x")\n')))
    assert "direct output print()" in problems and "no __all__" in problems


def test_docstrings_require_args_and_returns(tmp_path: Path) -> None:
    source = '''"""Doc."""
def f(a: int) -> int:
    """Summary."""
    return a
'''
    problems = "\n".join(check_docstrings.check_file(write(tmp_path, source)))
    assert "Args: missing a" in problems and "no Returns: section" in problems
