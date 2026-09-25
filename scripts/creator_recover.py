"""Recover only CPU video postprocessing for one already failed, settled job."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--job-directory', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'finalize'), required=True)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from training.demo.backend import prepare_postprocess_recovery, finalize_postprocess_recovery
    operation = prepare_postprocess_recovery if args.phase == 'prepare' else finalize_postprocess_recovery
    try:
        report = operation(args.job_directory, args.project_root)
    except Exception as error:
        print(json.dumps({'error': type(error).__name__, 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
