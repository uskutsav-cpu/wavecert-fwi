from wavecert.benchmarks.procedural import GEOLOGY_SETTINGS


def test_phase9_has_six_named_setting_proxies():
    assert GEOLOGY_SETTINGS == (
        "penobscot",
        "f3",
        "gom",
        "fault",
        "salt_canopy",
        "seam",
    )
