import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from forecast import grouped_split, main, validate_splits
from models.heads.student_t import nll_student_t_mixture_2d
from utils.data import TrajJsonlDataset


class ForecastTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, records):
        path = self.root / "sample.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return path

    def test_bundled_schema_normalizes_once(self):
        path = self.write([dict(tid=7, hist=[[1, .5, .25, .1, .1]], fut=[[2, .75, .5, .1, .1]])])
        data = TrajJsonlDataset(path)
        np.testing.assert_allclose(data[0]["past_xy"], [[0, -.5]])
        np.testing.assert_allclose(data[0]["future_xy"], [[.5, 0]])
        self.assertEqual(data.items[0]["track_id"], 7)
        with self.assertRaisesRegex(ValueError, "not calibrated pixels"):
            TrajJsonlDataset(path, return_pixels=True)

    def test_pixel_schema_and_validation(self):
        record = dict(past=[[50, 25]], future=[[75, 50]], W=100, H=100, track_id=1)
        path = self.write([record])
        np.testing.assert_allclose(TrajJsonlDataset(path)[0]["past_xy"], [[0, -.5]])
        del record["W"]
        with self.assertRaisesRegex(ValueError, "positive W and H"):
            TrajJsonlDataset(self.write([record]))
        with self.assertRaisesRegex(ValueError, "no trajectory records"):
            TrajJsonlDataset(self.write([]))

    def test_grouped_split_is_disjoint_and_repeatable(self):
        records = [dict(tid=i // 2, hist=[[1, .5, .5, .1, .1]], fut=[[2, .5, .5, .1, .1]]) for i in range(20)]
        data = TrajJsonlDataset(self.write(records))
        split, unit = grouped_split(data)
        self.assertEqual(unit, "track")
        self.assertEqual(split, grouped_split(data)[0])
        groups = [{data.items[i]["track_id"] for i in ids} for ids in split.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sorted(i for ids in split.values() for i in ids), list(range(20)))
        validate_splits(data, split, unit)
        corrupt = dict(split, test=split["train"][:1])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_splits(data, corrupt, unit)

    def test_video_groups_and_bad_records(self):
        records = [dict(video=f"v{i // 2}", track_id=i % 2, past=[[1, 1]], future=[[1, 1]], W=2, H=2) for i in range(12)]
        data = TrajJsonlDataset(self.write(records))
        splits, unit = grouped_split(data)
        self.assertEqual(unit, "video")
        validate_splits(data, splits, unit)
        for invalid in ([1, 2], dict(records[0], past=[[float("nan"), 1]]), dict(records[0], video=None)):
            with self.assertRaises(ValueError):
                TrajJsonlDataset(self.write([invalid]))

    def test_student_t_loss_matches_torch_distribution(self):
        torch.manual_seed(3)
        mu = torch.randn(2, 3, 4, 2, requires_grad=True)
        log_scale = torch.randn(2, 3, 4, 2) * .2
        logits = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 2)
        dof = torch.tensor(5.)
        component = torch.distributions.StudentT(dof, mu, log_scale.exp()).log_prob(target.unsqueeze(2)).sum(-1)
        expected = -torch.logsumexp(logits.log_softmax(-1) + component, -1).mean()
        actual = nll_student_t_mixture_2d(logits, mu, log_scale, dof, target)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(mu.grad).all())

    def test_train_checkpoint_reload_and_provenance(self):
        records = [dict(tid=i, hist=[[1, .5, .5, .1, .1]], fut=[[2, .51, .52, .1, .1]]) for i in range(6)]
        path = self.write(records)
        run = self.root / "run"
        main(["--data", str(path), "--smoke", "--out", str(run)])
        first = json.loads((run / "metrics.json").read_text())
        main(["--data", str(path), "--checkpoint", str(run / "best.pt"), "--out", str(self.root / "eval")])
        second = json.loads((self.root / "eval/metrics.json").read_text())
        self.assertEqual(first["test"], second["test"])
        records[0]["tid"] = 20
        self.write(records)
        with self.assertRaisesRegex(ValueError, "Dataset differs"):
            main(["--data", str(path), "--checkpoint", str(run / "best.pt")])


if __name__ == "__main__":
    unittest.main()
