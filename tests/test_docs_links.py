"""Every relative link and image in the public docs points to a file that exists."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CONTRIBUTING.md", ROOT / "docs/research/README.md"]
DOCS += sorted((ROOT / "docs").glob("*.md")) + sorted(ROOT.glob("*/README.md"))
LINK = re.compile(r'\]\(([^)#\s]+)(?:#[^)]*)?\)|src="([^"]+)"')


def test_relative_links_resolve():
    broken = [
        f"{doc.relative_to(ROOT)} -> {target}"
        for doc in DOCS
        for m in LINK.finditer(doc.read_text())
        if not (target := m.group(1) or m.group(2)).startswith("http") and not (doc.parent / target).exists()
    ]
    assert not broken, broken
