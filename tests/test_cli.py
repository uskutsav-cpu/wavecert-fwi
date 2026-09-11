from wavecert.cli import build_parser


def test_phase4_6_cli_subcommands_are_registered():
    parser = build_parser()
    cases = {
        "certificate-study": ["certificate-study", "--cases", "1"],
        "calibrate-certificate": ["calibrate-certificate", "--alpha", "0.1"],
        "adaptive-study": ["adaptive-study", "--cases", "1"],
    }

    for command, argv in cases.items():
        args = parser.parse_args(argv)
        assert args.command == command
