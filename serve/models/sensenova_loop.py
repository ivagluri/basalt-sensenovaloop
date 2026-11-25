"""The SenseNova Loop dialect: the Qwen one (the model's Jinja chat template, `<tool_call><function=...>` calls, the
Qwen3-VL video text), with what its template does not have taken out: no reasoning-effort levels (thinking is on or off,
`enable_thinking`) and no switch that keeps earlier turns' thinking."""
from __future__ import annotations

from serve.models.qwen import Qwen


class SenseNovaLoop(Qwen):
    name = "sensenova_loop"
    effort_turn = ""
    # chat_template.jinja: tools and several calls per turn, a leading system turn, thinking on or off
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": False,
            "supports_system_role": True, "supports_preserve_reasoning": False}

    def template_kwargs(self, kwargs: dict) -> dict:
        out = dict(kwargs)
        out.pop("preserve_reasoning", None)
        return out
