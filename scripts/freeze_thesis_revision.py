"""Write or verify a local thesis snapshot without committing the user's worktree."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/thesis-revision-v2-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources() -> None:
    for generic in sorted((ROOT / "paper/chapters").glob("*.tex")):
        school = ROOT / "paper-csust/body" / generic.name
        normalized = school.read_text().replace(r"\chap{", r"\chapter{").replace(
            r"\subsect{", r"\subsection{").replace(r"\sect{", r"\section{")
        if generic.read_text() != normalized:
            raise ValueError(f"chapter versions diverged: {generic.name}")
    generic = (ROOT / "paper/main.tex").read_text()
    school = (ROOT / "paper-csust/body/abstract.tex").read_text()
    for start, stop in [("金属热处理中的炉温", r"\keywords"),
                        ("Furnace temperature can", r"\par\noindent")]:
        body = school[school.index(start):school.index(stop, school.index(start))].strip()
        if body not in generic:
            raise ValueError("abstract versions diverged")
    for built, exported in [("paper/build/main.pdf", "output/pdf/heat-treatment-world-model-thesis-draft.pdf"),
                            ("paper-csust/build/thesis.pdf", "output/pdf/heat-treatment-world-model-thesis-csust.pdf")]:
        if sha256(ROOT / built) != sha256(ROOT / exported):
            raise ValueError(f"export is stale: {exported}")


def collect_files(manifest: Path = MANIFEST) -> list[Path]:
    files = set()
    for folder in ["src", "tests", "scripts", "docs", "paper", "paper-csust"]:
        for path in (ROOT / folder).rglob("*"):
            if (path.is_file() and path != manifest and "build" not in path.parts
                    and "__pycache__" not in path.parts and path.name != ".DS_Store"):
                files.add(path)
    for name in ["README.md", "pyproject.toml", "uv.lock", "requirements.txt",
                 "output/pdf/heat-treatment-world-model-thesis-draft.pdf",
                 "output/pdf/heat-treatment-world-model-thesis-csust.pdf"]:
        files.add(ROOT / name)
    for relative in re.findall(r"\| `([^`]+)` \| `[a-f0-9]{64}` \|",
                               (ROOT / "docs/experiment-freeze.md").read_text()):
        files.add(ROOT / relative)
    # Include the result pool behind older single-seed tables and figure analyses,
    # as well as the designated main-claim files above.
    for suffix in ("*.json", "*.csv", "*.png"):
        files.update((ROOT / "outputs").rglob(suffix))
    for suffix in ("*.py", "*.log", "*.tex"):
        files.update((ROOT / "outputs/c45_audit_followup_20261001").rglob(suffix))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write a new snapshot only after paper QA")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="Snapshot path; use a new path to preserve a historical revision")
    parser.add_argument("--revision", default=None,
                        help="Revision name for a new snapshot")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    check_sources()
    if args.write:
        inputs = [ROOT / "outputs/c45_cross_solver/cross_solver_dataset.npz"]
        inputs.extend(ROOT / f"outputs/c45_effective_boundary/seed_{seed}/weight_0p001.pt"
                      for seed in (42, 7, 123))
        if args.revision and args.revision != "thesis-revision-v2-20260905":
            for directory in ["outputs/c45_audit_followup_20261001",
                              "outputs/c45_effective_boundary", "outputs/c45_ood_partial_observability"]:
                for extension in ("*.pt", "*.npz", "*.npy"):
                    inputs.extend((ROOT / directory).rglob(extension))
            for directory in ["outputs/c45_physics_weight_sweep", "outputs/c45_physics_weight_seed7",
                              "outputs/c45_physics_weight_seed123"]:
                inputs.extend(ROOT / directory / name for name in ("weight_0.pt", "weight_0p001.pt"))
        inputs = sorted(set(inputs))
        prior_revision = (json.loads(manifest.read_text())["revision"]
                          if manifest.exists() else "thesis-revision-v2-20260905")
        payload = {
            "revision": args.revision or prior_revision,
            "kind": "local content snapshot; not a Git tag or public release",
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "includes_uncommitted_work": True,
            "files": {str(p.relative_to(ROOT)): sha256(p) for p in collect_files(manifest)},
            "local_binary_inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        }
        canonical = json.dumps({k: payload[k] for k in ("files", "local_binary_inputs")}, sort_keys=True).encode()
        payload["content_snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    payload = json.loads(manifest.read_text())
    missing, changed = [], []
    for section in ("files", "local_binary_inputs"):
        for name, expected in payload[section].items():
            path = ROOT / name
            if not path.exists():
                missing.append(name)
            elif sha256(path) != expected:
                changed.append(name)
    unexpected = sorted(str(p.relative_to(ROOT)) for p in collect_files(manifest)
                        if str(p.relative_to(ROOT)) not in payload["files"])
    if missing or changed or unexpected:
        raise SystemExit(json.dumps({"missing": missing, "changed": changed, "unrecorded": unexpected}, indent=2))
    print(f"Verified {len(payload['files'])} files and {len(payload['local_binary_inputs'])} local inputs; chapters, abstracts and PDF exports agree.")
    print(payload["content_snapshot_sha256"])


if __name__ == "__main__":
    main()
