from __future__ import annotations

import json
from pathlib import Path


def test_all_tutorial_notebooks_are_valid_nonempty_v4_documents():
    notebooks = sorted(Path("tutorials").glob("*.ipynb"))
    assert len(notebooks) == 8
    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["nbformat"] == 4
        assert isinstance(document["cells"], list) and document["cells"]
        for cell in document["cells"]:
            assert cell["cell_type"] in {"markdown", "code", "raw"}
            assert isinstance(cell.get("source"), list)
