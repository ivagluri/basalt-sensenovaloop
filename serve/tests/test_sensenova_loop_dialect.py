"""The SenseNova Loop dialect: the architecture picks it, its template renders thinking on and off as the checkpoint's own
chat_template.jinja does, and what that template lacks (effort levels, the switch that keeps earlier thinking) is neither
advertised nor passed on. Text only: no engine, no pack."""
import unittest

from serve.models import dialect_for
from serve.models.qwen import Qwen
from serve.models.sensenova_loop import SenseNovaLoop

# the generation prompt of the checkpoint's template (sensenova/SenseNova-Loop-9B, chat_template.jinja), trimmed to what
# these checks read
TEMPLATE = """{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}"""


class SenseNovaLoopDialectTest(unittest.TestCase):
    def test_picked_by_architecture(self):
        d = dialect_for("sensenova_loop")
        self.assertIsInstance(d, SenseNovaLoop)
        self.assertIsInstance(d, Qwen)

    def test_caps(self):
        d = SenseNovaLoop()
        self.assertFalse(d.caps["supports_reasoning_effort"])
        self.assertFalse(d.caps["supports_preserve_reasoning"])
        self.assertTrue(d.caps["supports_tool_calls"])
        self.assertEqual(d.effort_turn, "")

    def test_template_kwargs_drop_preserve(self):
        d = SenseNovaLoop()
        self.assertEqual(d.template_kwargs({"preserve_reasoning": True, "enable_thinking": False}), {"enable_thinking": False})

    def test_thinking_switch(self):
        t = SenseNovaLoop().template(TEMPLATE)
        msgs = [{"role": "user", "content": "hi"}]
        on = t.render(msgs, add_generation_prompt=True)
        off = t.render(msgs, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(on.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))


if __name__ == "__main__":
    unittest.main()
