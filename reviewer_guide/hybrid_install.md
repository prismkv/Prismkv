# Hybrid 27B dependency recipe for PPU Torch 2.4

The recipe uses Linux x86-64, CPython 3.10, the existing PPU SDK Torch 2.4.0
build, Transformers 5.8.0 and NumPy 1.26.4. Keep the base Torch environment.
Use a new dependency target so the ordinary Qwen3-8B environment and saved
model files remain separate.

The unmodified Transformers wheel installs successfully. Importing the
hybrid class on the tested Torch build fails during grouped-matmul operator
registration because that Torch release cannot infer a schema from postponed
annotations. The helper verifies the exact wheel source hash and supplies
the schema explicitly. The operator implementation, autograd registration
and model math are untouched.

Run from the repository root with your existing PPU SDK environment active.
The lock names the 29 Linux wheels resolved by pip. An existing wheelhouse
can use the same command with `--no-index --find-links /path/to/wheelhouse`.

```bash
deps="$PWD/.prismkv-hybrid-deps"
test ! -e "$deps" || exit 1
python -m pip install --ignore-installed --no-deps --only-binary=:all: \
  --index-url https://pypi.org/simple --require-hashes --target "$deps" \
  -r reviewer_guide/hybrid_runtime/requirements-linux-cp310.lock
python reviewer_guide/hybrid_runtime/patch_hybrid_torch24_schema.py "$deps"
export PYTHONPATH="$PWD/src:$deps"
export PYTHONNOUSERSITE=1
python -m unittest discover -s tests -v
```

The fresh wheel install and patched hybrid CPU validation used the staged
vendor Torch build. The same source passed the 24-test ordinary CPU suite
under both Transformers 4.57.1 and patched 5.8.0 without skips. A four-layer
hybrid CPU fixture checked causal prefill, cached logits, exact greedy output,
sparse routing on both standard-attention groups and incremental updates.
The public loader now retains the checkpoint's rotary base when explicitly
enabling or disabling YaRN on the newer config API.

See [the CPU and lock validation record](hybrid_runtime/validation.json) and
[the hybrid model CLI](hybrid_reference.md). The patch is specific to the
tested Torch 2.4/Transformers 5.8.0 combination. These checks do not measure
native latency, serving behavior or full LongBench/RULER quality.
