"""Optional pytest plugin: a terminal summary of every sigeval verdict in a run.

You never need this — assert_eval works in a plain pytest suite. But when a
suite has many eval cases, a rolled-up table ("3 PASS, 1 INCONCLUSIVE, 1 FAIL")
is what a reviewer actually wants to see. Enable via:

    # conftest.py
    pytest_plugins = ["sigeval.pytest_plugin"]

Then call sigeval.record(result) after assert_eval / run_case, or use the
record()-wrapped assert_eval below.
"""

_COLLECTED = []


def record(result):
    """Register a CaseResult (or list of them) for the end-of-run summary."""
    if isinstance(result, (list, tuple)):
        _COLLECTED.extend(result)
    else:
        _COLLECTED.append(result)
    return result


def pytest_terminal_summary(terminalreporter):
    if not _COLLECTED:
        return
    tr = terminalreporter
    tr.write_sep("=", "sigeval")
    counts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
    for r in _COLLECTED:
        counts[r.verdict.name] += 1
        tr.write_line(f"  {r.verdict.name:<12} {r.case_id:<24} {r.verdict}")
    tr.write_line(
        f"  -> {counts['PASS']} PASS, {counts['INCONCLUSIVE']} INCONCLUSIVE, "
        f"{counts['FAIL']} FAIL"
    )
    _COLLECTED.clear()
