"""Real tiny-backbone replay checks; not a downstream score."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
if importlib.util.find_spec('transformers') is not None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
else:
    Qwen3ForCausalLM = None
from prismkv.evaluation import evaluate_jsonl


@unittest.skipIf(Qwen3ForCausalLM is None, 'install optional transformers dependency')
class FullEvaluation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1207)
        config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=96,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=512, attention_dropout=0.,
            sliding_window=None, use_sliding_window=False)
        config._attn_implementation = 'sdpa'
        self.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        self.ids = torch.randint(0,128,(1,13))

    def test_jsonl_full_matches_native_generation(self):
        with torch.inference_mode():
            expected = self.model.generate(input_ids=self.ids, max_new_tokens=5,
                do_sample=False, eos_token_id=None, pad_token_id=0)[0,13:].tolist()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.jsonl'
            source.write_text(json.dumps(dict(id='case0', input_ids=self.ids[0].tolist(),
                max_new_tokens=5,eos_token_ids=[]))+'\n')
            done=evaluate_jsonl(self.model,source,root/'run',prefill_chunk_size=5)
            result=json.loads((root/'run/predictions.jsonl').read_text())
            self.assertTrue(done['complete'])
            self.assertEqual(result['generated_token_ids'],expected)
            self.assertEqual(result['termination'],'cap')
            self.assertEqual(len(result['self_generated_token_nll']),5)
            self.assertEqual(self.model.config._attn_implementation,'sdpa')


if __name__ == '__main__':
    unittest.main()
