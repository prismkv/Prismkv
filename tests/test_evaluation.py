"""Real tiny-backbone replay checks; not a downstream score."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
if importlib.util.find_spec('transformers') is not None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
else:
    Qwen3ForCausalLM = None
from prismkv.evaluation import evaluate_jsonl,run_request
from prismkv.core import SharedSAE
from prismkv.hf_reference import Encoder


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

    def encoders(self):
        return {(l,g):Encoder(SharedSAE(16,32,2,4),1.,1.) for l in range(2) for g in range(2)}

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

    def test_same_reference_ppl_matches_independent_full_logits(self):
        row=dict(id='ppl',input_ids=self.ids[0].tolist(),max_new_tokens=5,eos_token_ids=[])
        full=run_request(self.model,row,prefill_chunk_size=5)
        target=full['generated_token_ids']
        teacher=dict(row,reference_token_ids=target,reference_kind='full_generated')
        with torch.inference_mode():
            sequence=torch.tensor([row['input_ids']+target[:-1]])
            logits=self.model(sequence,use_cache=False).logits[0,12:17].float()
            independent=-logits.log_softmax(-1).gather(1,torch.tensor(target)[:,None]).flatten()
        routed=run_request(self.model,teacher,mode='all_read',encoders=self.encoders(),prefill_chunk_size=5)
        self.assertEqual(routed['generated_token_ids'],target)
        torch.testing.assert_close(torch.tensor(routed['reference']['token_nll']),independent,
                                   atol=3e-6,rtol=1e-5)
        self.assertAlmostEqual(routed['reference']['mean_nll'],float(independent.mean()),places=5)
        nll=routed['reference']['token_nll']
        self.assertEqual(routed['reference']['ppl'],math.exp(math.fsum(nll)/len(nll)))
        self.assertEqual(full['access']['source'],'derived_full_policy')
        self.assertEqual(routed['access']['source'],'instrumented_reference')
        for key in ['selected_old_unique','dense_generated_tail','group_steps']:
            self.assertEqual(full['access'][key],routed['access'][key])
        self.assertEqual(routed['access']['group_steps'],16)
        self.assertEqual(routed['access']['dense_generated_tail'],40)
        self.assertEqual(self.model.config._attn_implementation,'sdpa')

    def test_sparse_count_split_and_new_cache_reference(self):
        row=dict(id='sparse',input_ids=self.ids[0].tolist(),max_new_tokens=6,eos_token_ids=[])
        encoders=self.encoders()
        options=dict(mode='sparse',encoders=encoders,prefill_chunk_size=5,
                     selected_buckets=4,unique_cap=.25,update_interval=2)
        a=run_request(self.model,row,**options)
        b=run_request(self.model,dict(row,reference_kind='dataset_target',
                      reference_token_ids=[9,7,5,3]),**options)
        self.assertEqual(a['generated_token_ids'],b['generated_token_ids'])
        self.assertEqual(a['self_generated_token_nll'],b['self_generated_token_nll'])
        self.assertEqual(a['access'],b['access'])
        self.assertEqual(len(a['access_events']),20)
        self.assertTrue(any(r['history']>13 for r in a['access_events']))
        for event in a['access_events']:
            self.assertEqual(event['old_unique']+event['generated_indexed_unique'],event['unique'])
            self.assertEqual(event['old_raw']+event['generated_indexed_raw'],event['raw'])
            self.assertLessEqual(event['old_unique'],13)
            self.assertLessEqual(event['unique'],int(.25*event['history']))
            self.assertLessEqual(event['generated_indexed_unique'],event['history']-13)
            self.assertIn(event['tail'],[1,2])
        self.assertEqual(a['access']['selected_old_unique'],sum(r['old_unique'] for r in a['access_events']))
        self.assertEqual(b['reference']['termination'],'reference_length')
        self.assertEqual(len(b['reference']['token_nll']),4)
        self.assertEqual(self.model.config._attn_implementation,'sdpa')

    def test_failed_partial_output_and_no_overwrite(self):
        valid=dict(id='first',input_ids=self.ids[0].tolist(),max_new_tokens=2,eos_token_ids=[])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.jsonl'
            source.write_text(json.dumps(valid)+'\n'+json.dumps(valid)+'\n')
            with self.assertRaisesRegex(ValueError,'unique'):
                evaluate_jsonl(self.model,source,root/'run')
            self.assertFalse((root/'run/DONE.json').exists())
            failure=json.loads((root/'run/FAILED.json').read_text())
            self.assertEqual(failure['completed_requests'],1)
            preserved=(root/'run/predictions.jsonl').read_bytes()
            self.assertEqual(len(preserved.splitlines()),1)
            with self.assertRaises(FileExistsError):
                evaluate_jsonl(self.model,source,root/'run')
            self.assertEqual((root/'run/predictions.jsonl').read_bytes(),preserved)


if __name__ == '__main__':
    unittest.main()
