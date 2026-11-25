# SenseNova-Loop-9B (in progress)

A dense model with a loop. Runs with the same command as any model: `basalt serve PACK`. Status: it converts, loads and
serves (chat, tools, thinking, the MTP drafter) on one card, from the BF16 checkpoint (Q8_0) or a ModelOpt NVFP4 one;
there is no published pack, and the quality numbers below are a first look (two chunks). What is checked so far is below.

## Pack

Build it from the BF16 checkpoint ([sensenova/SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B),
Apache-2.0):

```sh
python3 tools/convert/convert.py --checkpoint SenseNova-Loop-9B --head q8 --name "SenseNova-Loop-9B Q8_0" \
    --out SenseNova-Loop-9B-Q8_0.basalt
```

- Every projection (GDN, attention, MLP, the MTP layer and with `--head q8` the head) Q8_0 from BF16; without `--head`
  the head becomes FP8 rows with a scale per row.
- The hyper-connection routers FP32 as stored; GDN a / b and the token embedding BF16; norms F32 with the offset folded in.
- The loop, the hyper-connections and the YaRN rope from `config.json`, under the official GGUF's key names
  (`sensenova_loop.loop.*`, `sensenova_loop.mhc.*`, `sensenova_loop.rope.scaling.*`). `loop_gate` is not used at a fixed pass
  count and is left out.
- The vision tower as for the 27B. 10.9 GiB; the embedding (1.9 GiB) stays in page-locked RAM.

## Architecture

The `sensenova_loop` architecture (code in `src/models/sensenova_loop`), on Basalt's shared dense stack (`src/dense`):

- **Layers:** 32 stored, hidden 4096, FFN 12288; 3 gated delta-net layers to 1 gated attention layer (16 query heads, 4 KV
  heads of 256; GDN 16 key and 32 value heads of 128).
- **Loop:** layers 0 to 3, then 4 to 27 twice, then 28 to 31: 56 layer visits a token. Each visit keeps its own KV cells (14
  attention visits) and recurrent state (42 GDN visits); the weights are shared. `dense::Geometry::physical` holds the plan.
- **Hyper-connections:** two residual streams over layers 4 to 27, opened from the residual at layer 4 and closed into their
  mean at layer 27 on each visit. Each sublayer's FP32 router reads the RMS-normalised streams and gives the branch weights,
  the output weights and a 2x2 mix from 20 rounds of log-space Sinkhorn (`kernels::hc2_pre`, `hc2_merge`, with the
  reference's BF16 roundings).
- **Rope:** interleaved M-RoPE over 64 of the 256 head dims, YaRN factor 2 over an original 262,144 at every position, as
  the checkpoint is configured: up to 524,288 positions.
- **Tokenizer:** the Qwen vocabulary with Qwen2 / Qwen3's pre-tokenizer split (`qwen2`: combining marks are not joined to
  letters), as the checkpoint's `tokenizer.json` gives it.

## Cards

One card. A second card in either `gpu-mode` is refused: the loop's layers run twice, so a cut or a prompt stage would
need some layers on both cards.

## Drafting

The checkpoint's MTP layer, on by default with `--spec 3`. Each MTP draft reads the whole head, and past the second position
few drafts are kept; on an RTX 5070 Ti (Q8_0 pack, greedy, 128 tokens): `--spec 3` 1.43 s, `--spec 4` 1.48 s, `--spec 2`
1.63 s, `--spec 0` 2.74 s, `--spec 16` 2.76 s. Drafted output is plain decoding's, token for token. The official DFlash 2
drafter (`sensenova/SenseNova-Loop-9B-DFlash2`) reads the final norm's output as a fifth feature and has a full-attention
last layer; neither is in the block drafter yet.

## What is checked

On an RTX 5070 Ti, the Q8_0 pack:

- Greedy tokens against the BF16 reference (transformers 5.18 and the checkpoint's own remote code, on the CPU): the first
  generated tokens match; where they part, the reference's top two were tied in BF16, and given Basalt's token the
  reference's next choices are Basalt's.
- `--kv fp16` and `--kv int8` give the same tokens on prompts of 1,500 and 4,756 tokens (two prompt chunks); `--kv nvfp4` the
  same at 4,756, a different continuation from the fourth token at 1,500.
- A prompt reads at about 4,000 tokens/s; decoding without drafting at about 47 tokens/s (Q8_0 reads about 13.8 GB a token);
  with the MTP drafter 55 to 100 tokens/s in the server.
- The server: answers, a tool call, thinking on and off.

## Quality against BF16

The base is the reference model itself: transformers 5.18 with the checkpoint's remote code, FP32 arithmetic on the BF16
weights (as llama.cpp's CPU path computes a BF16 GGUF), wikitext-2 test in 512-token chunks, the second half scored, written
in llama.cpp's KL-divergence base format. llama.cpp's port (pull request 32588) is not used: at its current head it gives a
looped attention layer one KV cache for both visits. Two chunks (510 rows), so the numbers are a first look, not the gate's
100:

| Pack | Size | On the card | PPL (base 11.922) | Mean KLD | Same top-1 |
| --- | --- | --- | --- | --- | --- |
| Q8_0 from BF16 (`--head q8`) | 10.9 GiB | 8.6 GiB | 11.912 | 0.0032 | 98.24% |
| Kaleto's NVFP4-MLP ([Kaleto/SenseNova-Loop-9B-NVFP4-MLP](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4-MLP)), converted as it is | 8.2 GiB | 5.4 GiB | 11.956 | 0.0905 | 87.45% |
| Kaleto's NVFP4 ([Kaleto/SenseNova-Loop-9B-NVFP4](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4)), converted as it is | 7.2 GiB | 4.2 GiB | 12.097 | 0.1396 | 86.67% |

The Q8_0 pack is within Q8_0's own rounding of the reference. The NVFP4 builds lose what their quantization loses (Kaleto
reports 0.094 and 0.153 to 0.160 mean KLD for them on their own method); both have an NVFP4 head. Both convert with the
same command (`--checkpoint` on the ModelOpt checkpoint). A better NVFP4 pack (a BF16 head as FP8 rows, calibration over
both visits of the loop) is not made yet.

Speed on the RTX 5070 Ti, 256 greedy tokens after a short prompt: Q8_0 45 tokens/s (`--spec 0`) and 89 (`--spec 3`); NVFP4
84 and 155; NVFP4-MLP 65 and 129. A 1,499-token prompt reads in 375 ms (Q8_0), 116 ms (NVFP4) and 193 ms (NVFP4-MLP).

Not yet: the 100-chunk gate, images and video against the reference, parallel slots.
