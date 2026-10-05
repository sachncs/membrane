"""Require complete Google-style docstrings throughout ``membrane/``.

Run ``python tools/check_docstrings.py`` (CI does). Stricter than
pydocstyle, which skips private names and optional sections:

* every module, class, and function (private and name-mangled ones
  included) has a docstring;
* a function with parameters (other than ``self``/``cls``) documents
  each of them under ``Args:`` (``*args``/``**kwargs`` by their names);
* a function that returns a value has ``Returns:``, and a generator has
  ``Yields:``.

Properties, ``@overload`` stubs, nested helper functions, and
generated protobuf modules are exempt from the section rules; nested
functions still need no docstring.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "membrane"
SECTION = re.compile(r"^\s*(Args|Arguments|Returns|Yields|Raises|Attributes|Example|Examples|Note|Notes):\s*$", re.M)


def sections(doc: str) -> dict[str, str]:
    """Split a Google docstring into its sections.

    Args:
        doc: The docstring text.

    Returns:
        dict[str, str]: Section name to body text (empty dict when there
        are no sections).
    """
    found: dict[str, str] = {}
    matches = list(SECTION.finditer(doc))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(doc)
        found[match.group(1)] = doc[match.end() : end]
    return found


def returns_value(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``func`` returns something other than ``None``.

    Args:
        func: The function definition.

    Returns:
        bool: True when the annotation is not ``None``/``NoReturn``, or
        (unannotated) when a ``return <value>`` occurs in its own body.
    """
    if func.returns is not None:
        text = ast.unparse(func.returns)
        return text not in {"None", "NoReturn", "Never", "typing.NoReturn"}
    return any(isinstance(node, ast.Return) and node.value is not None for node in walk_own(func))


def is_generator(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``func`` itself yields.

    Args:
        func: The function definition.

    Returns:
        bool: True when a ``yield`` occurs in its own body.
    """
    return any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in walk_own(func))


def walk_own(func: ast.AST) -> list[ast.AST]:
    """Walk a function's body without descending into nested scopes.

    Args:
        func: The function definition.

    Returns:
        list[ast.AST]: Nodes belonging to the function's own body.
    """
    out: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        out.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


def parameter_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Names a function's ``Args:`` section must document.

    Args:
        func: The function definition.

    Returns:
        list[str]: Parameter names, excluding ``self``/``cls`` and
        underscore-prefixed (unused) ones.
    """
    args = func.args
    names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return [n for n in names if n not in {"self", "cls"} and not n.startswith("_")]


def decorated_with(func: ast.FunctionDef | ast.AsyncFunctionDef, *names: str) -> bool:
    """Whether ``func`` carries one of the given decorators.

    Args:
        func: The function definition.
        *names: Decorator names (``property``, ``overload``, ...).

    Returns:
        bool: True when any decorator's final name is in ``names``.
    """
    for decorator in func.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        final = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if (
            final in names
            or final.endswith(".setter")
            or (isinstance(target, ast.Attribute) and target.attr in {"setter", "deleter"})
        ):
            return True
    return False


def check_function(rel: Path, func: ast.FunctionDef | ast.AsyncFunctionDef, problems: list[str]) -> None:
    """Check one function's docstring.

    Args:
        rel: File path relative to the repository.
        func: The function definition.
        problems: List the violations are appended to.
    """
    where = f"{rel}:{func.lineno}: {func.name}()"
    doc = ast.get_docstring(func)
    if not doc:
        problems.append(f"{where} has no docstring")
        return
    if decorated_with(func, "property", "overload", "cached_property"):
        return
    found = sections(doc)
    params = parameter_names(func)
    if params:
        documented = found.get("Args", "") + found.get("Arguments", "")
        missing = [p for p in params if not re.search(rf"^\s*\**{re.escape(p)}\b", documented, re.M)]
        if missing:
            problems.append(f"{where} Args: missing {', '.join(missing)}")
    if is_generator(func):
        if "Yields" not in found:
            problems.append(f"{where} has no Yields: section")
    elif returns_value(func) and "Returns" not in found and func.name != "__init__":
        problems.append(f"{where} has no Returns: section")


def check_file(path: Path) -> list[str]:
    """Return the docstring violations in one file.

    Args:
        path: A Python source file.

    Returns:
        list[str]: ``path:line: message`` entries.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(ROOT)
    problems: list[str] = []
    if not ast.get_docstring(tree):
        problems.append(f"{rel}:1: module has no docstring")

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    problems.append(f"{rel}:{node.lineno}: class {node.name} has no docstring")
                visit(node.body)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                check_function(rel, node, problems)

    visit(tree.body)
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
    sys.stderr.write(f"{len(problems)} docstring violation(s)\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
