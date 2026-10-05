"""Enforce Membrane's naming and output conventions.

Run ``python tools/check_naming.py`` (CI does). It fails when code under
``membrane/`` defines a *semi-private* name (one leading underscore):

* module-level variables, functions, and classes;
* class attributes and methods;
* instance attributes assigned as ``self._name``.

Names must be public, or name-mangled private (``__name``) inside a
class, and every module declares its public API in ``__all__``. Dunder names, the throwaway ``_``, function locals and
parameters, and generated protobuf modules are exempt.

It also fails on direct console output (``print``, ``console.print``,
``typer.echo``, ``rich.print``): everything goes through
:mod:`logging` (see :mod:`membrane.logging`).
"""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "membrane"
OUTPUT_CALLS = {("print",), ("console", "print"), ("typer", "echo"), ("rich", "print")}


def is_semi_private(name: str) -> bool:
    """Whether ``name`` uses the single-underscore convention.

    Args:
        name: An identifier.

    Returns:
        bool: True for ``_name``; False for ``name``, ``__name``,
        ``__dunder__`` and ``_``.
    """
    return name.startswith("_") and not name.startswith("__") and name != "_"


def target_names(node: ast.expr) -> Iterator[str]:
    """Yield plain names bound by an assignment target.

    Args:
        node: An assignment target.

    Yields:
        str: Each bound name.
    """
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from target_names(element)


def scope_definitions(body: list[ast.stmt]) -> Iterator[tuple[int, str]]:
    """Yield ``(line, name)`` for names defined directly in a module or class body.

    Args:
        body: Statements of the module or class.

    Yields:
        tuple[int, str]: Definition line and name.
    """
    for stmt in body:
        match stmt:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                yield stmt.lineno, name
            case ast.Assign(targets=targets):
                for target in targets:
                    for name in target_names(target):
                        yield stmt.lineno, name
            case ast.AnnAssign(target=target):
                for name in target_names(target):
                    yield stmt.lineno, name
            case ast.TypeAlias(name=ast.Name(id=name)):
                yield stmt.lineno, name


def self_attributes(cls: ast.ClassDef) -> Iterator[tuple[int, str]]:
    """Yield ``(line, name)`` for every ``self.<name> = ...`` in a class's methods.

    Args:
        cls: The class definition.

    Yields:
        tuple[int, str]: Assignment line and attribute name.
    """
    for node in ast.walk(cls):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                yield node.lineno, target.attr


def call_path(func: ast.expr) -> tuple[str, ...]:
    """Return the dotted path of a call target, e.g. ``("console", "print")``.

    Args:
        func: The called expression.

    Returns:
        tuple[str, ...]: Path components; empty when not a plain name chain.
    """
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return tuple(reversed(parts))
    return ()


def check_file(path: Path) -> list[str]:
    """Return the violations in one file.

    Args:
        path: A Python source file.

    Returns:
        list[str]: ``path:line: message`` entries.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(ROOT)
    problems: list[str] = []
    found: set[tuple[int, str]] = set()
    found.update((line, name) for line, name in scope_definitions(tree.body) if is_semi_private(name))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        found.update((line, name) for line, name in scope_definitions(cls.body) if is_semi_private(name))
        found.update((line, name) for line, name in self_attributes(cls) if is_semi_private(name))
    problems.extend(f"{rel}:{line}: semi-private name {name!r}" for line, name in sorted(found))
    defines_all = any(
        isinstance(stmt, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(t, ast.Name) and t.id == "__all__"
            for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
        )
        for stmt in tree.body
    )
    if not defines_all and path.name != "__main__.py":
        problems.append(f"{rel}:1: module has no __all__ (list its public API)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and call_path(node.func) in OUTPUT_CALLS:
            problems.append(f"{rel}:{node.lineno}: direct output {'.'.join(call_path(node.func))}(); use logging")
    return problems


def main() -> int:
    """Check every module under ``membrane/``.

    Returns:
        int: Process exit status (1 when violations exist).
    """
    problems: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "_pb2" in path.name:
            continue
        problems.extend(check_file(path))
    for problem in problems:
        sys.stderr.write(problem + "\n")
    sys.stderr.write(f"{len(problems)} naming/output violation(s)\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
