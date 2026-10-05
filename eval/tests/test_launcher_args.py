from pathlib import Path
import subprocess
import unittest

SCRIPT = Path(__file__).parents[1] / "sh/eval_single_node.sh"

class ArgumentsTests(unittest.TestCase):
    def test_invalid_arguments_fail_promptly(self):
        for option in ("--run_name", "--init_model_path", "--template", "--tp_size"):
            for suffix in ([], [""], ["--other"]):
                with self.subTest(option=option, suffix=suffix):
                    result = subprocess.run(["bash", str(SCRIPT), option] + suffix,
                        capture_output=True, text=True, timeout=2)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Missing value for " + option, result.stderr)
                    self.assertNotIn("convert_and_evaluate_gpu", result.stderr)

if __name__ == "__main__":
    unittest.main()
