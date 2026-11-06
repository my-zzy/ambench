# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the reconstructed OpenPI train entrypoint."""

import argparse
from pathlib import Path


def main() -> None:
    # Parse the optional supervisor directory; forward model arguments unchanged.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path)
    args, remaining = parser.parse_known_args()
    if args.run_dir is not None and "--help" not in remaining and "-h" not in remaining:
        from ambench_learn.policies.pi.training_log import run_logged

        raise SystemExit(run_logged(remaining, args.run_dir))
    from ambench_learn.policies.pi.runtime import main as run

    run("train", remaining)


if __name__ == "__main__":
    main()
