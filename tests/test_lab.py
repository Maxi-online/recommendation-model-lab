from pathlib import Path
import tempfile
import unittest
import numpy as np
from recommender_lab import Interaction, Popularity, BPRMatrixFactorization, temporal_split, synthetic_interactions, evaluate


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_interactions(7, users=12)
        self.train, self.validation, self.test = temporal_split(self.rows)

    def test_chronological_boundaries_and_no_pair_leakage(self):
        self.assertLess(max(r.timestamp for r in self.train), min(r.timestamp for r in self.validation))
        self.assertLess(max(r.timestamp for r in self.validation), min(r.timestamp for r in self.test))
        self.assertFalse({(r.user, r.item) for r in self.train} & {(r.user, r.item) for r in self.test})

    def test_training_vocab_and_seen_item_exclusion(self):
        model = BPRMatrixFactorization(seed=7, epochs=3).fit(self.train)
        user = self.train[0].user
        self.assertEqual(set(model.items), {r.item for r in self.train})
        self.assertFalse(set(model.recommend(user)) & model.seen[user])
        self.assertTrue(np.isfinite(model.user_factors).all())

    def test_seeded_training_and_safe_checkpoint_roundtrip(self):
        first = BPRMatrixFactorization(seed=19, epochs=3).fit(self.train)
        second = BPRMatrixFactorization(seed=19, epochs=3).fit(self.train)
        np.testing.assert_array_equal(first.user_factors, second.user_factors)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            first.save(path)
            restored = BPRMatrixFactorization.load(path)
            self.assertEqual(first.recommend(self.train[0].user), restored.recommend(self.train[0].user))

    def test_metrics_and_cold_start_are_explicit(self):
        model = Popularity().fit(self.train)
        metrics = evaluate(model, self.test + [Interaction('unseen-user', 'unseen-item', 99999)])
        self.assertEqual(metrics['cold_user_events_excluded'], 1)
        self.assertGreater(metrics['evaluated_users'], 0)
        self.assertTrue(0 <= metrics['recall_at_k'] <= 1)
        self.assertTrue(0 <= metrics['ndcg_at_k'] <= 1)
        self.assertEqual(model.recommend('new-user'), model.recommend('another-new-user'))

    def test_test_only_item_never_enters_training_catalog(self):
        model = BPRMatrixFactorization(epochs=1).fit(self.train)
        result = evaluate(model, [Interaction(self.train[0].user, 'test-only-item', 99999)])
        self.assertNotIn('test-only-item', model.items)
        self.assertEqual(result['cold_item_events_excluded'], 1)
        self.assertIsNone(result['recall_at_k'])


if __name__ == '__main__':unittest.main()
