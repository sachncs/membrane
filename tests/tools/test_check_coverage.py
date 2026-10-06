"""The coverage gate tool."""

import json

from tools import check_coverage


def report(total: float, **modules: float) -> dict:
    files = {name.replace("__", "/") + ".py": {"summary": {"percent_covered": p}} for name, p in modules.items()}
    return {"totals": {"percent_covered": total}, "files": files}


def test_passes_when_every_gate_holds(tmp_path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report(90.0, membrane__a=75.0, membrane__b=100.0)))
    assert check_coverage.main([str(path)]) == 0


def test_lists_every_miss(tmp_path, caplog) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report(80.0, membrane__a=69.9, membrane__b=100.0)))
    assert check_coverage.main([str(path)]) == 1
    assert "total coverage 80.0% is below 84%" in caplog.text
    assert "membrane/a.py: 69.9% is below 70%" in caplog.text
    assert "membrane/b.py" not in caplog.text
