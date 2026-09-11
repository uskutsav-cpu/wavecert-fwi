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
