# SenseNova-Loop-9B (in progress)

A dense model with a loop. Runs with the same command as any model: `basalt serve PACK`. Status: it converts, loads and
serves (chat, tools, thinking, the official DFlash 2 drafter or the MTP layer) on one card, from the BF16 checkpoint (Q8_0) or a ModelOpt NVFP4 one;
there is no published pack, and the quality numbers below are a first look (two chunks). What is checked so far is below.

## Pack

Build it from the BF16 checkpoint ([sensenova/SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B),
Apache-2.0):

```sh
python3 tools/convert/convert.py --checkpoint SenseNova-Loop-9B --drafter SenseNova-Loop-9B-DFlash2 --head q8 \
    --name "SenseNova-Loop-9B Q8_0 DFlash2" --out SenseNova-Loop-9B-Q8_0-DFlash2.basalt
```

`--drafter` adds the official block drafter ([sensenova/SenseNova-Loop-9B-DFlash2](https://huggingface.co/sensenova/SenseNova-Loop-9B-DFlash2),
Apache-2.0, measured against the checkpoint's revision `3cc220e9`) as `dflash.*` tensors; without it the pack drafts with
its MTP layer. 12.0 GiB with the drafter, 10.9 GiB without.

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

With a block drafter in the pack (`--drafter` above), the official DFlash 2 drafter, on by default with `--spec 8` (its
trained block). Five layers (967M parameters, its projections NVFP4 at load as for the 27B's drafter) read the target's
rows after layer visits 3, 27, 51 and 54 and, as a fifth feature, the target's final normed rows (the head's input); the
first four layers attend a 2,048-position window, the last the whole context, as it was trained (SenseSpec's per-layer
masks). LightLLM's integration windows every layer; on a long prompt the whole-context layer keeps about two more tokens a
window (below). Its context ring holds the whole context (about 0.5 GiB a slot at 128K cells); a checkpointed conversation
keeps its last 2,304 cells, so after a restore that layer reads as far back as the others until the conversation grows.

On an RTX 5070 Ti, Q8_0 pack, greedy, 256 tokens (`basalt generate`), output identical to `--spec 0` in every row, `--kv
fp16` and `int8` alike:

| Prompt | `--spec 0` | DFlash 2 `--spec 8` | tokens a window | MTP `--spec 3` |
| --- | ---: | ---: | ---: | ---: |
| a coding request (49 tokens) | 5.50 s | 1.56 s | 4.11 | 2.71 s |
| 1,500 tokens | 5.54 s | 1.92 s | 3.48 | |
| 4,756 tokens | 5.60 s | 1.22 s | 5.53 | 2.69 s |
| 4,756 tokens, the last layer windowed | | 2.02 s | 3.32 | |

`--spec 12` and `16` give the same times as 8; `--spec 6` is slower (1.65 s, 1.39 s). A draft costs about 2.7 ms, a verify
window about 22.5 ms (one plain token about 21.5 ms). With sampling (temperature 1.0) fewer drafts are kept.

Without a block drafter, the checkpoint's MTP layer, on by default with `--spec 3`. Each MTP draft reads the whole head, and
past the second position few drafts are kept; on an RTX 5070 Ti (Q8_0 pack, greedy, 128 tokens): `--spec 3` 1.43 s,
`--spec 4` 1.48 s, `--spec 2` 1.63 s, `--spec 0` 2.74 s, `--spec 16` 2.76 s. Drafted output is plain decoding's, token
for token.

## What is checked

On an RTX 5070 Ti, the Q8_0 pack:

- Greedy tokens against the BF16 reference (transformers 5.18 and the checkpoint's own remote code, on the CPU): the first
  generated tokens match; where they part, the reference's top two were tied in BF16, and given Basalt's token the
  reference's next choices are Basalt's.
- `--kv fp16` and `--kv int8` give the same tokens on prompts of 1,500 and 4,756 tokens (two prompt chunks); `--kv nvfp4` the
  same at 4,756 (400 tokens), a different continuation from the fourth token at 1,500, where the reference's top two are
  0.125 apart in log-probability (`" of"` -1.709, `" and"` -1.834) and nvfp4 takes the reference's first. On a prompt of
  21,851 tokens all three part within 50 tokens (fp16 and int8 at the 48th, fp16 and nvfp4 at the 36th), each on a sound
  continuation. KL divergence on the 2-chunk base: int8 0.0036 (same top-1 97.65%), nvfp4 0.0103 (95.49%).
- `--kv nvfp4` is what lets 131,072 cells fit beside the Q8_0 pack and the DFlash 2 drafter on a 16 GB card (3.1 GiB
  left); with int8 the planner stops about 235 MiB short.
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
