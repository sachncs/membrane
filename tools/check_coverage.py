"""Enforce the coverage gates on a ``coverage json`` report.

Run ``python tools/check_coverage.py coverage.json`` after a ``pytest
--cov=membrane --cov-report=json`` run (CI does). It fails when the total
is below :data:`MIN_TOTAL` percent or any module is below
:data:`MIN_MODULE` percent, and lists the offenders. Generated protobuf
modules are omitted in ``[tool.coverage.run]``.
"""

import json
import logging
import sys
from pathlib import Path

#: Required line coverage of the whole package, in percent.
MIN_TOTAL = 84.0
#: Required line coverage of every module, in percent.
MIN_MODULE = 70.0

logger = logging.getLogger("check_coverage")


def failures(report: dict) -> list[str]:
    """Return a message per gate the report misses.

    Args:
        report: Parsed ``coverage json`` output.

    Returns:
        list[str]: Empty when every gate holds.
    """
    problems = []
    total = report["totals"]["percent_covered"]
    if total < MIN_TOTAL:
        problems.append(f"total coverage {total:.1f}% is below {MIN_TOTAL:g}%")
    for path, data in sorted(report["files"].items()):
        percent = data["summary"]["percent_covered"]
        if percent < MIN_MODULE:
            problems.append(f"{path}: {percent:.1f}% is below {MIN_MODULE:g}%")
    return problems


def main(argv: list[str]) -> int:
    """Check the report named on the command line.

    Args:
        argv: ``[report_path]`` (default ``coverage.json``).

    Returns:
        int: 0 when every gate holds, 1 otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = json.loads(Path(argv[0] if argv else "coverage.json").read_text())
    problems = failures(report)
    for problem in problems:
        logger.error(problem)
    logger.info(
        "coverage: %.1f%% total, %d modules, %d below gate",
        report["totals"]["percent_covered"],
        len(report["files"]),
        len(problems),
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
