import json
from pathlib import Path
import tempfile
import unittest
import torch
from prismkv.core import SharedSAE
from prismkv.hf_reference import Encoder
from prismkv.weights import save_encoders,load_encoders


class BundleContracts(unittest.TestCase):
    def test_roundtrip_scales_and_rng(self):
        bundle = {(2,0):Encoder(SharedSAE(8,32,2,4),2.5,1.25)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'weights'
            save_encoders(bundle,root,metadata={'purpose':'unit test'})
            rng = torch.random.get_rng_state().clone()
            loaded, manifest = load_encoders(root)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            torch.testing.assert_close(loaded[2,0].sae.W_enc,bundle[2,0].sae.W_enc)
            self.assertEqual(loaded[2,0].key_scale,2.5)
            self.assertEqual(loaded[2,0].query_scale,1.25)
            self.assertEqual(manifest['metadata']['purpose'],'unit test')

    def test_corrupt_and_outside_files_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'weights'
            save_encoders({(0,0):Encoder(SharedSAE(4,16,2,4),1,1)},root)
            data = root/'l00_g00.pt'
            with data.open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaises(ValueError):load_encoders(root)
            manifest = json.loads((root/'manifest.json').read_text())
            manifest['groups'][0]['file'] = '../outside.pt'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_encoders(root)


if __name__=='__main__':unittest.main()
