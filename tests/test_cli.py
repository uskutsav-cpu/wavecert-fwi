from wavecert.cli import build_parser


def test_phase4_9_cli_subcommands_are_registered():
    parser = build_parser()
    cases = {
        "certificate-study": ["certificate-study", "--cases", "1"],
        "calibrate-certificate": ["calibrate-certificate", "--alpha", "0.1"],
        "adaptive-study": ["adaptive-study", "--cases", "1"],
        "end-to-end-study": ["end-to-end-study", "--cases", "1", "--iterations", "1"],
        "production-study": ["production-study", "--iterations", "1"],
        "ood-study": ["ood-study", "--cases-per-setting", "1"],
    }

    for command, argv in cases.items():
        args = parser.parse_args(argv)
        assert args.command == command


def test_end_to_end_cli_dispatches_verification_and_output(monkeypatch, capsys):
    from wavecert.cli import main

    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("wavecert.experiments.end_to_end.run_end_to_end_study", run)
    assert main([
        "end-to-end-study", "--cases", "2", "--iterations", "3", "--seed", "11",
        "--output", "custom-results", "--step-acceptance", "exact-armijo",
    ]) == 0
    assert captured == {
        "n_cases": 2, "iterations": 3, "seed": 11,
        "output_dir": "custom-results", "step_acceptance": "exact-armijo",
    }
    assert capsys.readouterr().out.strip() == "{}"


def test_evidence_cli_dispatches_complete_protocol_options(monkeypatch):
    from wavecert.cli import main

    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("wavecert.experiments.evidence_study.run_evidence_study", run)
    assert main([
        "evidence-study", "--checkpoint", "model.pt", "--output", "evidence-output",
        "--seeds", "13", "17", "--cases-per-seed", "2", "--iterations", "3",
        "--calibration-cases", "11", "--alpha", "0.2", "--scenarios", "clean", "frequency-shift",
        "--step-acceptance", "exact-armijo", "--threads", "2",
    ]) == 0
    assert captured["checkpoint_path"] == "model.pt"
    assert captured["output_dir"] == "evidence-output"
    config = captured["config"]
    assert config.seeds == (13, 17)
    assert config.cases_per_seed == 2 and config.iterations == 3 and config.calibration_cases == 11
    assert config.alpha == 0.2 and config.scenarios == ("clean", "frequency-shift")
    assert config.step_acceptance == "exact-armijo" and config.torch_threads == 2


def test_real_evaluation_and_devito_cli_dispatch_outputs(monkeypatch):
    from wavecert.cli import main

    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("wavecert.experiments.real_wavecert.run_real_wavecert_study", run)
    assert main([
        "real-wavecert-study", "--dataset", "fixture", "--checkpoint", "model.pt",
        "--alpha", "0.2", "--output", "evaluated",
    ]) == 0
    assert captured == {
        "dataset_dir": "fixture", "checkpoint_path": "model.pt", "alpha": 0.2,
        "output_dir": "evaluated",
    }
    captured.clear()
    monkeypatch.setattr("wavecert.experiments.devito_study.run_devito_study", run)
    assert main(["devito-study", "--marmousi", "model.bin", "--iterations", "2", "--output", "devito-out"]) == 0
    assert captured == {"marmousi_path": "model.bin", "optimizer_iterations": 2, "output_dir": "devito-out"}


def test_training_cli_dispatches_model_and_derivative_options(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    from wavecert.cli import main

    captured = {}
    threads = []

    def run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr("wavecert.experiments.real_geology_training.run_real_geology_training", run)
    assert main([
        "train-real-fno", "--dataset", "fixture", "--checkpoint", "trained.pt", "--output", "training",
        "--epochs", "7", "--batch-size", "8", "--patience", "3", "--width", "12", "--modes", "4",
        "--depth", "2", "--seed", "23", "--threads", "2", "--jacobian-weight", "0.1", "--jacobian-seed", "29",
    ]) == 0
    assert threads == [2]
    assert captured["dataset_dir"] == "fixture" and captured["checkpoint_path"] == "trained.pt"
    assert captured["output_dir"] == "training"
    assert (captured["epochs"], captured["batch_size"], captured["patience"]) == (7, 8, 3)
    assert captured["seed"] == 23 and captured["jacobian_seed"] == 29 and captured["jacobian_weight"] == 0.1
    assert captured["config"].width == 12 and captured["config"].depth == 2
    assert captured["config"].modes_z == captured["config"].modes_x == 4
