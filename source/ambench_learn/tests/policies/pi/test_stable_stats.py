# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Quaternion representation changes must not conceal real action errors."""

import unittest

import numpy as np
from openpi.policies import am_bench_policy as adapter

from ambench_learn.policies.pi.stable_stats import action_roundtrip_errors


class ActionRoundtripTests(unittest.TestCase):
    def test_nonunit_antipodal_targets_preserve_rotation(self):
        state = np.array([0.3, -0.1, 1.2, 1, 0, 0, 0, 0.1], dtype=np.float32)
        target = np.array([[0.4, 0.2, 1.3, -0.7071, -0.7071, 0, 0, 0.07]], dtype=np.float32)
        actual = adapter._to_ee_local_absolute(adapter._to_ee_local_relative(target, state), state)
        errors = action_roundtrip_errors(actual, target)
        self.assertGreater(errors["raw_component"], 1)
        for key in ("position_m", "unit_quaternion_component", "gripper"):
            self.assertLess(errors[key], 2e-6)
        actual[0, 0] += 0.02
        actual[0, 3:7] = [1, 0, 0, 0]
        errors = action_roundtrip_errors(actual, target)
        self.assertGreater(errors["position_m"], 0.019)
        self.assertGreater(errors["unit_quaternion_component"], 0.7)

    def test_invalid_quaternions_are_rejected(self):
        target = np.array([[0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.float32)
        for value in (0, 0.5, float("nan")):
            invalid = target.copy()
            invalid[0, 3] = value
            with self.assertRaises(ValueError):
                action_roundtrip_errors(target, invalid)
            with self.assertRaises(ValueError):
                action_roundtrip_errors(invalid, target)


if __name__ == "__main__":
    unittest.main()
