#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 yayoi
# SPDX-License-Identifier: MIT
"""Verify a pinned source-only package; optionally reconstruct an isolated tree.
Never patches, builds, tests or starts the actual worktree. A fresh --output is
required for reconstruction. Main reviews source.diff and integrates separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / "manifest.json").read_text())
    for entry in manifest["package_files"]:
        f = package / entry["path"]
        require(f.is_file() and len(f.read_bytes()) == entry["bytes"] and sha(f.read_bytes()) == entry["sha256"],
                "package bytes changed: " + entry["path"])
    baseline = manifest["production_source_snapshot"]
    if args.check_root:
        root = args.check_root.resolve()
        found = {str(f.relative_to(root)) for directory in ("src", "include", "cmake")
                 for f in (root / directory).rglob("*") if f.is_file()}
        found.add("CMakeLists.txt")
        require(found == {entry["path"] for entry in baseline}, "actual source inventory changed; rebase required")
        for entry in baseline:
            raw = (root / entry["path"]).read_bytes()
            require(len(raw) == entry["bytes"] and sha(raw) == entry["sha256"], "actual base changed: " + entry["path"])
        for entry in manifest["changes"]:
            if entry["base_sha256"] is None:
                require(not (root / entry["path"]).exists(), "new path already exists: " + entry["path"])
    if args.output:
        target = args.output.resolve()
        require(not target.exists(), "--output must be a new isolated directory")
        if args.check_root:
            require(not target.is_relative_to(args.check_root.resolve()), "reconstruct outside the actual source tree")
        require(not target.is_relative_to(package), "reconstruct outside this immutable package")
        shutil.copytree(package / "base", target)
        for entry in manifest["changes"]:
            destination = target / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package / "files" / entry["path"], destination)
    print("source package verified; no build, test or runtime qualification")

if __name__ == "__main__":
    main()
