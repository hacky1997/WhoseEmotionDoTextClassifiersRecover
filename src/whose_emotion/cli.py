"""Command-line interface: python run.py --stage <name> [--local] [--work-dir <folder>]."""

import argparse

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="whose-emotion",
        description="Run one stage of the study. README.md lists the stages and their order.",
    )
    parser.add_argument("--stage", required=True, help="stage name, for example audit, main, or analysis")
    parser.add_argument("--local", action="store_true",
                        help="run outside Colab and Kaggle and read secrets from environment variables")
    parser.add_argument("--work-dir", default=None, help="local working folder (default: ./work)")
    args = parser.parse_args()
    run(stage=args.stage, local=args.local, work_dir=args.work_dir)


if __name__ == "__main__":
    main()
