# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the reconstructed OpenPI serve entrypoint."""

from ambench_learn.policies.pi.runtime import main

if __name__ == "__main__":
    main("serve")
