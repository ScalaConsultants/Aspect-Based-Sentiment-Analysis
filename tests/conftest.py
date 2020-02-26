import pytest


def pytest_addoption(parser):
    parser.addoption("--run-sanity_checks", action="store_true",
                     default=False, help="Run sanity checks")


def pytest_configure(config):
    dsc = "sanity_check: mark test as sanity_check, which can be slow to run"
    config.addinivalue_line("markers", dsc)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-sanity_checks"):
        # --run-sanity-checks given in cli: do not skip sanity checks
        return
    reason_desc = "need --run-sanity_checks option to run"
    skip_sanity_check = pytest.mark.skip(reason=reason_desc)
    for item in items:
        if "sanity_check" in item.keywords:
            item.add_marker(skip_sanity_check)
