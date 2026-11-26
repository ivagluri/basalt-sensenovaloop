# basalt-sensenovaloop

A fork of [Basalt](https://github.com/jesdga95/basalt) that adds
[SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B): a dense 9B model whose middle 24 layers run twice
(56 layer passes a token) with two-stream hyper-connections, plus its official
[DFlash 2 drafter](https://huggingface.co/sensenova/SenseNova-Loop-9B-DFlash2). It is an experiment, not part of Basalt and
not supported by its maintainers. The model's own project, with its paper, code and evaluations:
[OpenSenseNova/SenseNova-Loop](https://github.com/OpenSenseNova/SenseNova-Loop). Everything else in Basalt is unchanged; this fork is rebased onto Basalt's `main` from
time to time.

## What it runs

On one RTX 5070 Ti (16 GB), the Q8_0 pack with the DFlash 2 drafter and an NVFP4 KV cache, at a context of 131,072 tokens
(3.1 GiB of VRAM left):

| | Plain decode | DFlash 2 drafting |
| --- | ---: | ---: |
| A short coding request, greedy | 47 tokens/s | 164 tokens/s |
| A 4,756-token prompt, greedy | 46 tokens/s | 210 tokens/s |
| An agent coding task (pi, sampled at temperature 1.0) | | 133 tokens/s over 108K generated tokens |

Drafted output is plain decoding's, token for token. A prompt reads at about 4,000 tokens/s. The model reads about 14 GB
of weights a token at Q8_0, about 86% of the card's bandwidth in the products: as a dense model it is slower per token
than the mixture-of-experts models Basalt runs, by design.

Quality against the BF16 model (transformers with the checkpoint's own code, FP32 arithmetic; wikitext-2, two chunks):

| Pack, KV cache | Mean KLD | Same top token |
| --- | ---: | ---: |
| Q8_0, int8 | 0.0036 | 97.65% |
| Q8_0, NVFP4 | 0.0103 | 95.49% |
| Kaleto's NVFP4-MLP | 0.0905 | 87.45% |
| Kaleto's NVFP4 | 0.1396 | 86.67% |

A semver range matcher written by the model through an agent harness passed 111, 112 and 113 of 113 hidden tests in three
runs (MTP drafting, DFlash 2 with int8 KV, DFlash 2 with NVFP4 KV); one sampled run each, so these are level.

## Quick start

```sh
git clone https://github.com/ivagluri/basalt-sensenovaloop && cd basalt-sensenovaloop
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=sensenova_loop
cmake --build build -j
pip install -r requirements.txt

hf download sensenova/SenseNova-Loop-9B --revision 3cc220e9c14bd7a74813c1299765e22147b1a3d5 --local-dir models/SenseNova-Loop-9B
hf download sensenova/SenseNova-Loop-9B-DFlash2 --local-dir models/SenseNova-Loop-9B-DFlash2
python3 tools/convert/convert.py --checkpoint models/SenseNova-Loop-9B --drafter models/SenseNova-Loop-9B-DFlash2 \
    --head q8 --name "SenseNova-Loop-9B Q8_0 DFlash2" --out models/SenseNova-Loop-9B-Q8_0-DFlash2.basalt
```

The conversion takes a couple of minutes and gives a 12.0 GiB pack (the token embedding, 1.9 GiB of it, stays in system
RAM). Then `basalt_config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/SenseNova-Loop-9B-Q8_0-DFlash2.basalt", "kv": "nvfp4", "max-context": 131072 }],
  "server": { "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

With `"kv": "int8"` a 16 GB card holds about 98,304 cells beside the drafter. Kaleto's ModelOpt NVFP4 checkpoints
([NVFP4](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4), [NVFP4-MLP](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4-MLP))
convert with the same command (without `--head`): smaller and faster, at the quality above.

The model guide, with the architecture, the drafter and every measurement: [docs/models/sensenova-loop.md](docs/models/sensenova-loop.md).

## Limits

- One GPU: the loop runs some layers twice, so Basalt's two-card modes are refused.
- Images and video use the 27B's vision path and are not checked against the reference; `--parallel` slots are untested.
- After a conversation is restored from a checkpoint, the drafter's whole-context layer sees its last 2,304 positions until
  the conversation grows (output is unaffected; only how many drafts are kept).
- The quality numbers are two chunks, not Basalt's 100-chunk gate.

## What the fork changes in Basalt

Model code in `src/models/sensenova_loop`, `serve/models/sensenova_loop.py` and `tools/convert/sensenova_loop.py`. In
shared code, each change inert for the models Basalt ships:

- `src/dense`: a layer plan (layer passes over stored layers, each pass with its own KV and recurrent state), two-stream
  hyper-connections around each sublayer (`src/kernels/cuda/mhc2.cu`), a shared pack reader (`dense::PackReader`), and two
  block-drafter features (the target's final normed rows as a feature; layers that attend the whole context).
- GDN kernels for 32 value heads beside 48; `RopeSpec` tables of 128 pairs (a whole 256-wide head).
- The tokenizer's `qwen2` split (combining marks not joined to letters); the converter's BF16 checkpoints (Q8_0), F32
  tensors and YaRN keys.

## Licences and credits

Basalt is MIT. SenseNova-Loop-9B and its DFlash 2 drafter are Apache-2.0 (SenseTime / OpenSenseNova). The implementation
follows the checkpoint's own remote code, SenseNova's [SenseSpec](https://github.com/OpenSenseNova/SenseSpec) (the drafter's
training masks), LightLLM's DFlash 2 integration and SGLang pull request 47281.
