"""Record/check the exact P0 sources; never invokes a solver or edits a paper.

Run with --write to deliberately refresh the manifest after reviewing a change.
The default is read-only verification. Byte identity is not model equivalence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "experiments/specs/v1_source_manifest.json"
FIXED = (
    "experiments/specs/source_snapshots/Intercity_Operation_BHH_3.pdf",
    "experiments/specs/source_snapshots/BHH_CFS_V1_Numerical_Experiment_Runbook.md",
    "coordination/V1_MODEL_FREEZE.md",
    "coordination/V1_F_RUNTIME_CONTRACT.md",
    "coordination/V1_M_IMPLEMENTATION_PROFILE.md",
    "experiments/specs/equation_to_code_matrix.md",
    "experiments/specs/experiment_data_contract.json",
    "experiments/specs/v1_p0_capacity_design.json",
    "experiments/specs/v1_p0_decisions.json",
    "experiments/specs/v1_p0_preflight.py",
    "experiments/reports/v1_p0_data_preflight.json",
    "experiments/specs/freeze_v1_p0_sources.py",
    "research/theory/v1_p0_parameter_design.md",
    "research/theory/v1_direct_endpoint_contract.md",
    "experiments/specs/V1_P1_BOUNDED_PROTOCOL.md",
    "experiments/specs/v1_p1_budget.json",
    "experiments/specs/run_v1_p1.py",
    "experiments/specs/run_v1_p1_micro.py",
    "experiments/specs/plot_v1_p1.py",
)
PATTERNS = (
    "research/paper/final_manuscript/**/*.tex",
    "research/paper/final_manuscript/**/*.bib",
    "intercity_delivery/v1/*.py",
    "tests/test_v1*.py",
)


def source_paths(root=ROOT):
    paths = set(FIXED)
    for pattern in PATTERNS:
        paths.update(p.relative_to(root).as_posix() for p in root.glob(pattern))
    return sorted(paths)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def content_fingerprint(entries):
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return digest_bytes(canonical)


def build_manifest(root=ROOT):
    entries = {}
    for relative in source_paths(root):
        content = (root / relative).read_bytes()
        entries[relative] = {"sha256": digest_bytes(content), "size_bytes": len(content)}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                            cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "schema_version": "v1_p0_source_manifest/1",
        "git_head_at_freeze": revision,
        "working_tree_dirty_at_freeze": bool(status),
        "identity_scope": "Exact bytes of enumerated files, including uncommitted changes",
        "mathematical_authority": "research/paper/final_manuscript/main.tex and its TeX/Bib project",
        "provided_pdf": "experiments/specs/source_snapshots/Intercity_Operation_BHH_3.pdf",
        "runbook_requested_pdf": "Intercity_Operation_BHH (3)(1).pdf",
        "missing_requested_pdf": True,
        "source_relationship": "Provided (3) PDF core equations audited against the local source; "
                               "no claim of byte identity, identical compilation or identity with (3)(1)",
        "does_not_authorize_solving": True,
        "files": entries,
        "content_sha256": content_fingerprint(entries),
    }


def check_manifest(manifest, root=ROOT):
    errors = []
    if manifest.get("schema_version") != "v1_p0_source_manifest/1":
        return ["unsupported manifest schema"]
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        return ["missing source entries"]
    if manifest.get("content_sha256") != content_fingerprint(entries):
        errors.append("manifest content fingerprint differs")
    if set(entries) != set(source_paths(root)):
        errors.append("source inventory differs (added or missing files)")
    for relative, expected in entries.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"source outside project: {relative}")
            continue
        try:
            content = path.read_bytes()
        except OSError:
            errors.append(f"unreadable source: {relative}")
            continue
        actual = {"sha256": digest_bytes(content), "size_bytes": len(content)}
        if actual != expected:
            errors.append(f"source bytes differ: {relative}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Explicitly refresh the current P0 freeze")
    args = parser.parse_args(argv)
    if args.write:
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = check_manifest(manifest)
    print(json.dumps({"valid": not errors, "files": len(manifest["files"]),
                      "content_sha256": manifest["content_sha256"], "errors": errors}, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
