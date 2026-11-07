"""Optional real-backbone integration tests with tiny random CPU Qwen3 weights."""
import copy
import gc
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
if importlib.util.find_spec('transformers') is not None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
else:
    Qwen3Config = Qwen3ForCausalLM = None
from prismkv.core import SharedSAE
from prismkv.hf_reference import Encoder, HFReference, load_reference_model


@unittest.skipIf(Qwen3Config is None, "install optional transformers dependency")
class BackboneContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(103)
        config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=96,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=512, attention_dropout=0.0,
            sliding_window=None, use_sliding_window=False)
        config._attn_implementation = 'sdpa'
        self.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        self.ids = torch.randint(0, 128, (1, 13))

    def encoders(self):
        return {(l,g):Encoder(SharedSAE(16,32,2,4),1.,1.) for l in range(2) for g in range(2)}

    @torch.no_grad()
    def test_dense_equivalence_and_backend_restoration(self):
        native, routed = self.model, copy.deepcopy(self.model)
        session = HFReference(routed, self.encoders(), mode='all_read')
        a = native(self.ids, use_cache=True)
        b = routed(self.ids, use_cache=True)
        torch.testing.assert_close(a.logits, b.logits, atol=1e-6, rtol=1e-5)
        for _ in range(4):
            token = a.logits[:,-1:].argmax(-1)
            a = native(token, past_key_values=a.past_key_values, use_cache=True)
            b = routed(token, past_key_values=b.past_key_values, use_cache=True)
            torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=1e-5)
        self.assertEqual(len(session.records), 4*2*2)
        session.close()
        self.assertEqual(routed.config._attn_implementation, 'sdpa')

    @torch.no_grad()
    def test_sparse_all_groups_budget_tail_and_append(self):
        routed = copy.deepcopy(self.model)
        session = HFReference(routed, self.encoders(), unique_cap=.25,
                              selected_buckets=4, update_interval=2)
        result = routed(self.ids, use_cache=True)
        for _ in range(5):
            token = result.logits[:,-1:].argmax(-1)
            result = routed(token, past_key_values=result.past_key_values, use_cache=True)
            self.assertTrue(torch.isfinite(result.logits).all())
        self.assertEqual(set(session.indices), set(self.encoders()))
        self.assertTrue(all(x.history==17 for x in session.indices.values()))
        self.assertTrue(all(r['unique'] <= int(.25*r['history']) for r in session.records))
        self.assertTrue(all(1 <= r['tail'] <= 2 for r in session.records))
        session.close()

    @torch.no_grad()
    def test_offset_chunked_prefill_and_missing_encoder(self):
        with self.assertRaises(ValueError):
            HFReference(copy.deepcopy(self.model), {})
        native, routed = self.model, copy.deepcopy(self.model)
        session = HFReference(routed, self.encoders(), mode='all_read')
        a = native(self.ids, use_cache=True)
        b = None
        with session.prefill():
            for start in range(0, 13, 5):
                b = routed(self.ids[:,start:start+5], past_key_values=None if b is None else b.past_key_values, use_cache=True)
        torch.testing.assert_close(a.logits[:,-1], b.logits[:,-1], atol=2e-6, rtol=1e-5)
        token = a.logits[:,-1:].argmax(-1)
        a = native(token, past_key_values=a.past_key_values, use_cache=True)
        b = routed(token, past_key_values=b.past_key_values, use_cache=True)
        torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=1e-5)
        session.close()

    @torch.no_grad()
    def test_generate_one_token_prefix(self):
        native,routed=self.model,copy.deepcopy(self.model)
        session=HFReference(routed,self.encoders(),mode='all_read')
        kwargs=dict(max_new_tokens=4,do_sample=False,eos_token_id=None,pad_token_id=0)
        a=native.generate(input_ids=self.ids[:,:1],**kwargs)
        b=session.generate(input_ids=self.ids[:,:1],**kwargs)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(session.prefix_lengths,{0:1,1:1})
        self.assertEqual(len(session.records),3*2*2)
        session.close()

    @torch.no_grad()
    def test_checkpoint_backend_does_not_force_flash_attention(self):
        # A real disk round trip reproduces the private backend field found in
        # pretrained configs, without depending on FA2 or network access.
        with tempfile.TemporaryDirectory() as folder:
            self.model.save_pretrained(folder)
            path = Path(folder)/'config.json'
            config = json.loads(path.read_text())
            config['_attn_implementation'] = 'flash_attention_2'
            path.write_text(json.dumps(config))
            original_bytes = path.read_bytes()
            loaded = load_reference_model(folder, dtype=torch.float32)
            self.assertEqual(loaded.config._attn_implementation, 'sdpa')
            self.assertEqual(path.read_bytes(), original_bytes)
            torch.testing.assert_close(self.model(self.ids).logits,
                                       loaded(self.ids).logits, atol=2e-6, rtol=1e-5)
            # Release safetensors mmap before removing a shared-filesystem tmpdir.
            del loaded
            gc.collect()

    @torch.no_grad()
    def test_eager_prefill_keeps_causality(self):
        native = copy.deepcopy(self.model)
        native.config._attn_implementation = 'eager'
        routed = copy.deepcopy(native)
        session = HFReference(routed, self.encoders(), mode='all_read')
        a = native(self.ids, use_cache=True)
        b = routed(self.ids, use_cache=True)
        torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=1e-5)
        token = a.logits[:,-1:].argmax(-1)
        a = native(token, past_key_values=a.past_key_values, use_cache=True)
        b = routed(token, past_key_values=b.past_key_values, use_cache=True)
        torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=1e-5)
        session.close()
        self.assertEqual(routed.config._attn_implementation, 'eager')


if __name__=='__main__':
    unittest.main()
