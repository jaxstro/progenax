"""Execute every python code block in README.md (audit R7/D3: would have caught
the deleted-API examples automatically).

Skips blocks marked ``# doctest: +SKIP`` (e.g. GPU-only or illustrative
fragments). Each block runs in a fresh namespace.
"""
import re
from pathlib import Path

import pytest

README = (Path(__file__).resolve().parents[2] / "README.md").read_text()
BLOCKS = [
    b for b in re.findall(r"```python\n(.*?)```", README, re.DOTALL)
    if "+SKIP" not in b
]


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_readme_block_executes(i):
    exec(compile(BLOCKS[i], f"README.md:block{i}", "exec"), {})
