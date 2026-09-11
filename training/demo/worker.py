"""Operator-launched actual GPU worker, with mandatory reservation/authorization."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-directory', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    from training.demo.backend import run_model_job
    run_model_job(args.job_directory, args.project_root, args.max_seconds)


if __name__ == '__main__':
    main()
