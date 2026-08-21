"""REQ-2.2 / AC-5: vendored files are byte-identical to source except the
two-line provenance header. Body hashes are recorded at vendor time."""
import hashlib
import json
from pathlib import Path

VENDOR_DIR = Path(__file__).parent.parent / "rlbot" / "vendor"
CHECKSUMS = json.loads((Path(__file__).parent / "vendor_checksums.json").read_text())

HEADER_LINES = 2


def test_all_vendored_modules_match_recorded_source_hash():
    for name, expected in CHECKSUMS.items():
        path = VENDOR_DIR / f"{name}.py"
        assert path.exists(), f"vendored module missing: {path}"
        lines = path.read_text().splitlines(keepends=True)
        assert lines[0].startswith("# VENDORED from"), f"{name}: missing provenance header"
        body = "".join(lines[HEADER_LINES:])
        actual = hashlib.sha256(body.encode()).hexdigest()
        assert actual == expected, f"{name}: vendored body drifted from recorded source hash"


def test_no_unrecorded_vendored_files():
    py_files = {p.stem for p in VENDOR_DIR.glob("*.py")} - {"__init__"}
    assert py_files == set(CHECKSUMS), (
        f"vendor dir contents {py_files} != recorded {set(CHECKSUMS)}"
    )
