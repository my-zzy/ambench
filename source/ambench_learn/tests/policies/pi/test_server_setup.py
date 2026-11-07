# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Guard against training on incomplete assets or accidentally selecting fake data."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from flax import nnx

from ambench_learn.policies.pi.config import make_config
from ambench_learn.policies.pi.runtime import main


class ServerSetupTests(unittest.TestCase):
    def test_multitask_separates_artifacts_and_requires_request_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = make_config(work_dir=root, corrected=True)
            multi = make_config(work_dir=root, multitask=True, finetune_mode="full")
            multi_stats = make_config(work_dir=root, multitask=True)
            self.assertNotEqual(multi.assets_dirs, single.assets_dirs)
            self.assertNotEqual(multi.checkpoint_dir, single.checkpoint_dir)
            self.assertEqual(Path(multi.data.assets.assets_dir), multi_stats.assets_dirs)
            self.assertTrue(multi.data.base_config.prompt_from_task)
            self.assertTrue(multi.policy_metadata["requires_prompt"])
            self.assertNotIn("task_prompt", multi.policy_metadata)
            self.assertEqual(single.policy_metadata["task_prompt"], "press the button")

    def test_full_tuning_reuses_statistics_and_separates_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = make_config(work_dir=root, corrected=True)
            full = make_config(work_dir=root, corrected=True, finetune_mode="full", steps=20_000, fsdp_devices=2)
            self.assertEqual(Path(full.data.assets.assets_dir), pilot.assets_dirs)
            self.assertNotEqual(full.checkpoint_dir, pilot.checkpoint_dir)
            self.assertIs(full.freeze_filter, nnx.Nothing)
            self.assertEqual(full.model.paligemma_variant, "gemma_2b")
            self.assertEqual(full.model.action_expert_variant, "gemma_300m")
            self.assertEqual(full.lr_schedule.decay_steps, 20_000)
            self.assertEqual(pilot.lr_schedule.decay_steps, 10_000)
            self.assertEqual(pilot.model.paligemma_variant, "gemma_2b_lora")

    def test_synthetic_data_cannot_enter_real_training(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main("train", ["--synthetic-data"])
        self.assertEqual(error.exception.code, 2)

    def test_incomplete_transfer_fails_before_creating_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Complete and verify the artifact transfer"):
                main("train", ["--work-dir", directory, "--variant", "corrected", "--finetune-mode", "full"])
            self.assertFalse((Path(directory) / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
