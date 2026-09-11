from wavecert.physics.production import devito_environment_report


def test_devito_environment_report_is_explicit():
    report = devito_environment_report()
    assert "devito_installed" in report
    assert "executable" in report
    assert isinstance(report["executable"], bool)
