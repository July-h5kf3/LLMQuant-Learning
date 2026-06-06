import ast
from pathlib import Path
from unittest import TestCase, main


class QIGWAInputTest(TestCase):
    def test_wa_quant_without_distort_has_q_input_fallback(self):
        root = Path(__file__).resolve().parents[1]
        source = root.joinpath("qmllm/methods/qig/quantize/pre_quant.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        q_input_names = [
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword)
            and node.arg == "q_input"
            and isinstance(node.value, ast.Name)
        ]
        self.assertIn("wa_q_input", q_input_names)

        fallback_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "wa_q_input" for target in node.targets)
        ]
        self.assertTrue(
            any(isinstance(node.value, ast.Name) and node.value.id == "layer_input" for node in fallback_assignments),
            "W4A8 QIG must pass the current layer input as q_input when distort is disabled.",
        )

        layer_input_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "layer_input" for target in node.targets)
            and isinstance(node.value, ast.Name)
            and node.value.id == "inps"
        ]
        layer_forward_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "inps" for target in node.targets)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Call)
            and isinstance(node.value.value.func, ast.Name)
            and node.value.value.func.id == "layer"
        ]
        self.assertTrue(layer_input_lines)
        self.assertTrue(layer_forward_lines)
        self.assertLess(
            min(layer_input_lines),
            min(layer_forward_lines),
            "layer_input must be captured before inps is updated to the next layer input.",
        )


if __name__ == "__main__":
    main()
