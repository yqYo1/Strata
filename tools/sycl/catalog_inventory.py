#!/usr/bin/env python3
"""Record header provenance and index Intel API names from an icpx JSON AST.

This records declarations, not B570 support. See the catalog for commands.
Uses only the Python standard library; does not contact the network or GPU.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def api_names(path):
    source = path.read_text()
    decoder = json.JSONDecoder()
    offset = 0
    names = {}

    def visit(node, scope):
        kind, name = node.get("kind"), node.get("name", "")
        if kind == "NamespaceDecl":
            if name in ("detail", "native") or name.startswith("__"):
                return
            scope = (*scope, name)
        if kind in ("FunctionDecl", "FunctionTemplateDecl"):
            if name and not name.startswith(("_", "<")):
                names.setdefault("::".join(scope), set()).add(name)
        # Do not index local functions, expressions, or implementation bodies.
        if kind in ("NamespaceDecl", "FunctionTemplateDecl", "LinkageSpecDecl"):
            for child in node.get("inner", []):
                visit(child, scope)

    while offset < len(source):
        while offset < len(source) and source[offset].isspace():
            offset += 1
        if offset == len(source):
            break
        node, offset = decoder.raw_decode(source, offset)
        visit(node, ())
    return {k: sorted(v) for k, v in sorted(names.items())}


def extension_index(spec_root, revision, macro_file, output):
    macros = dict(re.findall(r"^#define (SYCL_EXT_\w+) (.*)$", macro_file.read_text(), re.M))
    (output / "FEATURE_MACROS.json").write_text(json.dumps(macros, sort_keys=True, indent=2) + "\n")
    text = ["# Extension specification coverage index", "",
            f"Upstream Intel LLVM revision: `{revision}`; inspected 2026-10-02.",
            "All named extension/API documents in the upstream supported, experimental,",
            "proposed, deprecated and removed directories are included. This includes",
            "non-GPU and non-computational extensions to make exclusions explicit.",
            "The macro column comes from the installed compiler preprocessing `api_index.cpp`.",
            "A nonzero macro does not prove B570 support. A missing macro does not prove",
            "all related declarations absent (for example `dot_acc`). Referenced dependency",
            "macros are also included, so this is not a per-extension implementation verdict.",
            "See [runtime results](B570_CAPABILITIES.txt) and [the operation catalog](../SYCL_BATTLEMAGE_OPERATIONS.md)",
            "for B570 semantics, restrictions, exclusions and execution evidence.", "",
            "| Upstream state | Specification/API document | Installed referenced macros |",
            "| --- | --- | --- |"]
    manifest = []
    for file in sorted(spec_root.rglob("sycl_ext_*")):
        if file.suffix not in (".md", ".asciidoc") or "examples" in file.parts:
            continue
        path = file.relative_to(spec_root).as_posix()
        keys = sorted(set(re.findall(r"\bSYCL_EXT_[A-Z0-9_]+\b", file.read_text(errors="replace"))))
        values = ", ".join(f"`{k}={macros[k]}`" for k in keys if k in macros) or "None defined"
        url = f"https://github.com/intel/llvm/blob/{revision}/sycl/doc/extensions/{path}"
        text.append(f"| {path.split('/')[0]} | [{file.stem}]({url}) | {values} |")
        manifest.append({"path": path, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                         "referenced_macros": keys})
    text += ["", f"Coverage count: {len(manifest)} named extension/API documents.",
             "Header-only facilities and Khronos extensions are covered separately in the catalog.", ""]
    (output / "EXTENSION_INDEX.md").write_text("\n".join(text))
    (output / "EXTENSION_SOURCES.json").write_text(json.dumps({"revision": revision, "files": manifest}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-root", action="append", type=Path, required=True)
    parser.add_argument("--intel-ast", type=Path)
    parser.add_argument("--spec-root", type=Path)
    parser.add_argument("--spec-revision")
    parser.add_argument("--feature-macros", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.spec_root and not (args.spec_revision and args.feature_macros):
        parser.error("--spec-root requires --spec-revision and --feature-macros")
    args.output.mkdir(parents=True, exist_ok=True)
    lines = ["include_root\trelative_path\tsha256"]
    for root in args.include_root:
        if not root.is_dir():
            parser.error(f"missing include root: {root}")
        for file in sorted(root.rglob("*")):
            if file.is_file():
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                lines.append(f"{root.resolve()}\t{file.relative_to(root)}\t{digest}")
    (args.output / "INSTALLED_HEADERS.tsv").write_text("\n".join(lines) + "\n")
    if args.intel_ast:
        names = api_names(args.intel_ast)
        (args.output / "INTEL_API_NAMES.json").write_text(json.dumps(names, indent=2) + "\n")
        text = ["# Installed Intel namespace function-name index", "",
                "Generated from the host-side icpx 2026.1.1 AST. This is a declaration",
                "index, **not** a GPU support list. Overloads are collapsed; class members,",
                "`detail`, `native`, underscored names and function bodies are excluded.",
                "Some visible helper functions remain. Headers compiled for this index and",
                "the reproduction command are listed in [the catalog](../SYCL_BATTLEMAGE_OPERATIONS.md).",
                "Use that catalog for B570 restrictions and execution evidence.", ""]
        for namespace, functions in names.items():
            text += [f"## sycl::ext::{namespace} ({len(functions)} names)", "", "```text"]
            # One name per line keeps diffs and searches unambiguous.
            text.extend(functions)
            text += ["```", ""]
        (args.output / "INTEL_API_NAMES.md").write_text("\n".join(text))
    if args.spec_root:
        extension_index(args.spec_root, args.spec_revision, args.feature_macros, args.output)
    print(f"Indexed {len(lines) - 1} header-tree files.")


if __name__ == "__main__":
    main()
