def pytest_addoption(parser):
    parser.addoption(
        "--run-notebook",
        action="store_true",
        default=False,
        help="Re-execute the tutorial notebook from scratch before running checks.",
    )
