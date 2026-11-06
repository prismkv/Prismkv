import unittest
import torch
from prismkv.core import (SharedSAE, positive_global_topk, summarize, build_index,
                          bucket_scores, select_buckets, attend, append_index)
from prismkv.training import TrainingBatch, objective, mine_pairs


class ReferenceContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_global_budget_variable_cardinality_and_zero(self):
        a = torch.tensor([[10., 9., 8., 7.], [1., .9, .8, .7]])
        ids = positive_global_topk(a, 4)
        self.assertEqual((ids // 4).tolist(), [0, 0, 0, 0])
        self.assertEqual(len(positive_global_topk(torch.zeros_like(a), 4)), 0)
        self.assertEqual(positive_global_topk(torch.ones(2, 4), 3).tolist(), [0, 1, 2])

    def test_global_and_query_budget_are_different(self):
        model = SharedSAE(2, 4, 1, 2)
        with torch.no_grad():
            model.W_enc.copy_(torch.tensor([[4., 0.], [3., 0.], [2., 0.], [1., 0.]]))
        code, _ = model.encode(torch.tensor([[1., 0.], [10., 0.], [.01, 0.]]), 1)
        self.assertEqual((code > 0).sum(-1).tolist(), [2, 2, 0])

    def test_moments_of_original_keys_and_empty_bucket(self):
        k = torch.tensor([[2., 4.], [4., 8.], [9., 0.]])
        idx = summarize(k, [torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([], dtype=torch.long)])
        torch.testing.assert_close(idx.mean[0], torch.tensor([3., 6.]))
        torch.testing.assert_close(idx.variance[0], torch.tensor([1., 4.]))
        self.assertEqual(idx.counts.tolist(), [2, 1, 0])
        score, _ = bucket_scores(torch.ones(1, 2), idx)
        self.assertEqual(score[-1].item(), 0)

    def test_gqa_normalize_before_aggregation(self):
        k = torch.tensor([[10., 0.], [0., 2.], [-3., -1.]])
        idx = summarize(k, [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])])
        q = torch.tensor([[1., 0.], [0., 1.]])
        score, _ = bucket_scores(q, idx, terms=("mean",), query_topk=3)
        wanted = (q @ k.T / (2 ** .5)).softmax(-1).mean(0)
        torch.testing.assert_close(score, wanted)
        self.assertGreater((wanted - (q @ k.T / (2 ** .5)).mean(0).softmax(-1)).abs().max().item(), .01)

    def test_overlap_unique_budget_skips_overflow(self):
        keys = torch.zeros(10, 2)
        idx = summarize(keys, [torch.tensor([0, 1, 2, 3]), torch.tensor([2, 3]), torch.tensor([4])])
        ids, meta = select_buckets(torch.ones(1, 2), idx, unique_cap=.3, query_topk=3, terms=("count",))
        self.assertEqual(ids.tolist(), [2, 3, 4])
        self.assertEqual(meta["features"], [1, 2])
        self.assertEqual(meta["raw"], 3)
        full_ids, meta = select_buckets(torch.ones(1, 2), idx, query_topk=2, terms=("count",))
        self.assertEqual(full_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(meta["raw"], 6)

    def test_full_and_selected_original_attention_with_dense_tail(self):
        k, v, q = torch.randn(9, 4), torch.randn(9, 4), torch.randn(3, 4)
        result = attend(q, k, v, torch.arange(6), 6)
        torch.testing.assert_close(result, (q @ k.T / 2).softmax(-1) @ v)
        ids = torch.tensor([1, 3])
        chosen = torch.tensor([1, 3, 6, 7, 8])
        torch.testing.assert_close(attend(q, k, v, ids, 6), (q @ k[chosen].T / 2).softmax(-1) @ v[chosen])
        with self.assertRaises(ValueError):
            attend(q, k, v, torch.tensor([1, 1]), 6)

    def test_empty_index_and_zero_budget(self):
        k = torch.randn(5, 2)
        idx = summarize(k, [torch.tensor([], dtype=torch.long)])
        ids, _ = select_buckets(torch.ones(1, 2), idx)
        self.assertEqual(len(ids), 0)
        idx = summarize(k, [torch.arange(5)])
        ids, _ = select_buckets(torch.ones(1, 2), idx, unique_cap=0)
        self.assertEqual(len(ids), 0)

    def test_incremental_moments_preserve_old_memberships(self):
        k, new = torch.randn(17, 4), torch.randn(7, 4) + 4
        sae = SharedSAE(4, 12, 2, 3)
        index = build_index(k, sae, 1)
        before = [x.clone() for x in index.lists]
        merged = append_index(index, new, sae, 1)
        wanted = summarize(torch.cat((k, new)), merged.lists, index.center)
        torch.testing.assert_close(merged.mean, wanted.mean, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(merged.variance, wanted.variance, atol=1e-5, rtol=1e-5)
        for old, new_ids in zip(before, merged.lists):
            self.assertTrue(torch.equal(new_ids[:len(old)], old))
        self.assertEqual(merged.memberships - index.memberships, 7 * 2)
        self.assertTrue(torch.equal(merged.center, index.center))

    def test_query_feature_shared_weighted_route(self):
        sae = SharedSAE(2, 4, 1, 1)
        with torch.no_grad():
            sae.W_enc.copy_(torch.tensor([[3., 0.], [0., 2.], [-2., 0.], [0., -1.]]))
        idx = summarize(torch.randn(4, 2), [torch.tensor([i]) for i in range(4)])
        q = torch.eye(2)
        score, order = bucket_scores(q, idx, router="q_activation", sae=sae, query_scale=1, query_topk=1)
        torch.testing.assert_close(score, torch.tensor([.5, .5, 0., 0.]))
        self.assertEqual(order.tolist(), [0, 1])

    def test_mining_and_loss_gradient(self):
        q, k = torch.randn(8, 4), torch.randn(60, 4)
        qi, pos, neg = mine_pairs(q, k, queries=4)
        order = (q[qi] @ k.T).argsort(dim=-1, descending=True)
        for j in range(4):
            self.assertIn(pos[j], order[j, :2])
            self.assertFalse(torch.isin(neg[j], order[j, :6]).any())
        sae = SharedSAE(4, 16, 2, 4)
        pairs = torch.cat((q[qi], k[pos], k[neg].flatten(0, 1)))
        batch = TrainingBatch(torch.cat((q, k[:8])), 8, pairs, 4)
        ema = torch.ones(16) / 16
        loss, metrics = objective(sae, batch, ema, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sae.W_enc.grad.norm().item(), 0)
        self.assertGreater(sae.W_dec.grad.norm().item(), 0)
        self.assertLessEqual(metrics["mean_key_activations"].item(), 2)


if __name__ == "__main__":
    unittest.main()
