"""SenseNova Loop's part of the qwen35 checkpoint converter (checkpoint_qwen35.py): the checkpoint is a qwen3_5 text model
with a loop over some of its layers and two-stream hyper-connections, so the qwen35 plan converts it and asks this module
for what differs: the routers' tensors, the tensor it leaves out, the loop and hyper-connection keys (the official GGUF's
names), the layer visits a drafter's target layers count in, and the tokenizer's split."""
import checkpoint as C
import model_sections as S

ARCH = "sensenova_loop"

# the hyper-connection routers of a layer (FP32 [8][2 n_embd] and [8])
LAYER = {
    "mhc_attn.weight": ("hc_attn_fn.weight", "F32", "id"),
    "mhc_attn.bias": ("hc_attn_fn.bias", "F32", "id"),
    "mhc_ffn.weight": ("hc_ffn_fn.weight", "F32", "id"),
    "mhc_ffn.bias": ("hc_ffn_fn.bias", "F32", "id"),
}
SKIP = {"model.language_model.loop_gate.logits"}   # the loop's gate: unused at a fixed pass count


def is_loop(ck: C.Checkpoint) -> bool:
    return ck.config.get("model_type") == ARCH


def is_router(gguf_name: str) -> bool:
    return gguf_name.startswith("hc_")


def routers(ck: C.Checkpoint) -> list:
    """The router tensors the engine reads: every layer the hyper-connections span."""
    mhc = ck.config["mhc"]
    return [f"blk.{l}.hc_{p}_fn.{w}" for l in range(int(mhc["start"]), int(mhc["end"]))
            for p in ("attn", "ffn") for w in ("weight", "bias")]


def visits(ck: C.Checkpoint) -> int:
    """The layer visits a token runs through: the stored layers, the loop's once more per extra pass."""
    loop = ck.config["loop_settings"]
    return int(ck.text["num_hidden_layers"]) + (int(loop["eval_count"]) - 1) * (int(loop["end"]) - int(loop["start"]))


def kv(ck: C.Checkpoint, A: str) -> dict:
    """The loop and hyper-connection keys, checked: a fixed pass count, two streams, the hyper-connections inside one phase
    of the loop."""
    c, tc = ck.config, ck.text
    n = int(tc["num_hidden_layers"])
    loop, mhc = c.get("loop_settings"), c.get("mhc")
    if not isinstance(loop, dict) or not isinstance(mhc, dict):
        raise S.ConvertError("config.json: a sensenova_loop checkpoint without loop_settings or mhc")
    start, end, count = (int(loop[k]) for k in ("start", "end", "eval_count"))
    if not (0 <= start < end <= n and count >= 1) or loop.get("counts") != [count]:
        raise S.ConvertError(f"config.json: loop_settings {loop}: a fixed pass count over layers inside the model only")
    ms, me, streams, iters = (int(mhc[k]) for k in ("start", "end", "streams", "iterations"))
    scale = float(mhc.get("pre_scale", 1.0))
    if streams != 2 or iters < 1 or mhc.get("fixed_identity") is not False or not (scale > 0) or \
            set(mhc) - {"start", "end", "streams", "iterations", "fixed_identity", "pre_scale"}:
        raise S.ConvertError(f"config.json: mhc {mhc}: two streams with a learnt mix only")
    if not any(a <= ms < me <= b for a, b in ((0, start), (start, end), (end, n))):
        raise S.ConvertError("config.json: the hyper-connections cross a phase of the loop")
    return {A + "loop.start": start, A + "loop.end": end, A + "loop.count": count,
            A + "mhc.start": ms, A + "mhc.end": me, A + "mhc.streams": streams, A + "mhc.iterations": iters,
            A + "mhc.fixed_identity": False, A + "mhc.pre_scale": C.f32(scale)}


def tokenizer_kv(ck: C.Checkpoint) -> dict:
    """Qwen2 / Qwen3's pre-tokenizer split, which the checkpoint's tokenizer.json keeps (combining marks not joined to letters)."""
    return C.tokenizer_kv(ck, "qwen2", C.QWEN2_HF_SPLIT)
