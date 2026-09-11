#!/usr/bin/env python3
"""Serve the localhost-only asynchronous trained demo; GPU disabled without an operator guard."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True, help='Operator-only JSON: rollout_config, project_root, jobs_root, optional guard_command')
    args = parser.parse_args()
    from training.demo.contracts import Deployment
    from training.demo.service import DemoService, make_server
    service = DemoService(Deployment.load(args.deployment))
    server = make_server(service)
    print(f'InterActWorld local demo: http://127.0.0.1:{server.server_address[1]}', flush=True)
    print('Generation enabled with operator guard.' if service.deployment.guard_command else 'Preview only: GPU generation is disabled.', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
