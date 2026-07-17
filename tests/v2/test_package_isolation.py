from __future__ import annotations

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).parents[2] / "audience_trend_miner"


class V2PackageIsolationTest(unittest.TestCase):
    def test_supported_package_contains_only_v2_runtime_modules(self) -> None:
        top_level_modules = {
            path.name
            for path in PACKAGE_ROOT.iterdir()
            if path.name != "__pycache__"
        }

        self.assertEqual(top_level_modules, {"__init__.py", "__main__.py", "v2"})

    def test_v2_runtime_imports_only_v2_application_modules(self) -> None:
        unsupported_imports: list[str] = []
        for source_path in (PACKAGE_ROOT / "v2").rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported_modules = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module is not None
                    else [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                for imported_module in imported_modules:
                    if imported_module.startswith(
                        "audience_trend_miner."
                    ) and not imported_module.startswith("audience_trend_miner.v2"):
                        unsupported_imports.append(
                            f"{source_path.relative_to(PACKAGE_ROOT)}: {imported_module}"
                        )

        self.assertEqual(unsupported_imports, [])


if __name__ == "__main__":
    unittest.main()
