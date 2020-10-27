import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true",
                     default=False, help="Run slow tests")


def pytest_configure(config):
    dsc = "slow: mark test, which can be slow to run"
    config.addinivalue_line("markers", dsc)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow checks
        return
    reason_desc = "need --run-slow option to run"
    skip_sanity_check = pytest.mark.skip(reason=reason_desc)
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_sanity_check)
