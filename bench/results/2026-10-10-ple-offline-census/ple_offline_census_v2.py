#!/usr/bin/env python3
"""Bounded CPU offline geometry. Never opens a model shard or runs the oracle.
Every independently computed row must equal the supplied original-C++ oracle.
No timings, real cache misses, PleReader bytes, physical I/O or inference claims.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

MAX_TOKENS = 262144
CHUNK = 8192
HEADS = 16
MAX_USES = CHUNK * HEADS
INPUT_BYTES = 8 * 1024 * 1024
TABLE_OFFSET, ROW_BYTES, NROWS = 192, 90, 320001536
TABLE_END = TABLE_OFFSET + ROW_BYTES * NROWS
PAGE, RECORD = 4096, 131072
EOS, MASK = 248044, (1 << 64) - 1
MULT = (23703573157769, 20109073645365, 8052911324071)
VOCAB = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
         20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
         20000153, 20000159, 20000161, 20000171)
OFFSET = (0, 20000003, 40000026, 60000059, 80000106, 100000165,
          120000228, 140000297, 160000374, 180000455, 200000548,
          220000655, 240000802, 260000955, 280001114, 300001275)
FIXTURE_SHA = {
    "32k": "137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449",
    "32k-insert198": "137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449",
    "256k": "cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a",
}
FROZEN_COMMIT = "495369cf4dc6e6da564095a1c26504dd3694e3f2"
SOURCE_FILES = ("src/kernels/ngram.cpp", "include/strata/kernels/ngram.hpp",
                "src/ngram/ple_reader.cpp", "src/prefill/prefill.cpp",
                "sycl/src/prefill/prefill.cpp", "sycl/src/core/session.cpp")
DEFAULT_ROOT = "/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010"
SOURCE_SHA = dict(zip(SOURCE_FILES, (
    "e609caf1bb0812f366d96a5304c0f98e2864c1c0d22a0bef7b35d818e09debd5",
    "edc6d36eaf6062e8c377577de787ed9e5576ede19e04c5964dc7e6b83b8d01ac",
    "ddec6dbcc8afd8186c724403d5aaa54b70b9fbd551fea733ce422974131ede9e",
    "c2f6249f855241ff06cb7441d9e11a380b29c9183ea96b84b74b5591503d1815",
    "f6cbc4acb9f84b19480a25892a2e6864896a3e3ab0f68ace3c3f51a788591f69",
    "939e49516eb8586b51bfd686261a2cae167ca3e1e9af375d00618940a7df9158")))


def checked_path(s):
    if not s or len(os.fsencode(s)) > 4096:
        raise ValueError("path empty or exceeds 4096 bytes")
    return Path(s)


def bounded_int(s):
    if not re.fullmatch(r"-?(0|[1-9][0-9]*)", s) or len(s) > 11:
        raise argparse.ArgumentTypeError("invalid decimal int32")
    n = int(s)
    if not -(1 << 31) <= n < (1 << 31):
        raise argparse.ArgumentTypeError("int32 overflow")
    return n


def digest_file(path, cap):
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            b = f.read(min(65536, cap + 1 - total))
            if not b:
                break
            total += len(b)
            if total > cap:
                raise ValueError("provenance file exceeds byte bound")
            h.update(b)
    return {"sha256": h.hexdigest(), "bytes": total}


def load_tokens(path, fixture):
    with path.open("rb") as f:
        raw = f.read(INPUT_BYTES + 1)
    if len(raw) > INPUT_BYTES:
        raise ValueError("token file byte bound exceeded")
    sha = hashlib.sha256(raw).hexdigest()
    if fixture != "tiny" and sha != FIXTURE_SHA[fixture]:
        raise ValueError("original fixture SHA256 mismatch")
    # Scan instead of split(): even malicious whitespace/short words stays bounded.
    tokens = []
    for match in re.finditer(rb"\S+", raw):
        if len(tokens) >= MAX_TOKENS + 1:
            raise ValueError("input token count exceeds bound")
        text = match.group().decode("ascii")
        tokens.append(bounded_int(text))
    if not tokens:
        raise ValueError("empty token fixture")
    original_count = len(tokens)
    if fixture in ("32k", "32k-insert198") and original_count != 32768:
        raise ValueError("32K fixture count mismatch")
    if fixture == "256k" and original_count != MAX_TOKENS + 1:
        raise ValueError("256K fixture count mismatch")
    if fixture == "32k-insert198":
        tokens.insert(32759, 198)
    return tokens, {"path": str(path), "sha256": sha, "bytes": len(raw),
                    "original_tokens": original_count, "transformed_tokens": len(tokens)}


def independent_rows(tokens, index, initial):
    # Independent translation: explicitly gather nearest then older predecessors.
    ctx = [tokens[index]]
    cut = False
    for distance in (1, 2):
        at = index - distance
        predecessor = tokens[at] if at >= 0 else initial[at + 2]
        if cut or predecessor < 0 or predecessor == EOS:
            cut = True
            ctx.append(EOS)
        else:
            ctx.append(predecessor)
    products = [((value & MASK) * mult) & MASK for value, mult in zip(ctx, MULT)]
    mixed2 = products[0] ^ products[1]
    mixed3 = mixed2 ^ products[2]
    return [(mixed2 if h < 8 else mixed3) % VOCAB[h] + OFFSET[h] for h in range(HEADS)]


def histogram(values):
    return {str(k): v for k, v in sorted(collections.Counter(values).items())}


def geometry(rows, start, count):
    if len(rows) != count * HEADS or len(rows) > MAX_USES:
        raise ValueError("chunk descriptor reconciliation failed")
    jobs = {}  # first aligned page -> [max length, use count, unique rows]
    unique_rows = set()
    row_records = set()
    row_straddlers = 0
    record_straddlers = 0
    for row in rows:
        if not 0 <= row < NROWS:
            raise ValueError("out-of-table row")
        unique_rows.add(row)
        at = TABLE_OFFSET + row * ROW_BYTES
        end = at + ROW_BYTES
        if end > TABLE_END:
            raise ValueError("row byte bound")
        first = at // PAGE * PAGE
        length = ((end - 1) // PAGE + 1) * PAGE - first
        if length not in (PAGE, 2 * PAGE):
            raise ValueError("unsupported page job length")
        row_straddlers += length == 2 * PAGE
        r0, r1 = at // RECORD, (end - 1) // RECORD
        row_records.add(r0)
        row_records.add(r1)
        record_straddlers += r0 != r1
        if first not in jobs:
            jobs[first] = [length, 0, set()]
        j = jobs[first]
        j[0] = max(j[0], length)
        j[1] += 1
        j[2].add(row)
    if sum(j[1] for j in jobs.values()) != len(rows):
        raise ValueError("job use coverage failure")
    # Expanded pages are unioned separately: adjacent 8K/4K jobs can overlap.
    pages = set()
    jobs_per_record = collections.Counter()
    baseline_bytes = 0
    beyond_eof_bytes = 0
    beyond_eof_jobs = 0
    for first, (length, _, _) in jobs.items():
        baseline_bytes += length
        beyond_eof_bytes += max(0, first + length - TABLE_END)
        beyond_eof_jobs += first + length > TABLE_END
        for page in range(first // PAGE, (first + length) // PAGE):
            pages.add(page)
        for record in range(first // RECORD, (first + length - 1) // RECORD + 1):
            jobs_per_record[record] += 1
    occupied = collections.Counter(page // (RECORD // PAGE) for page in pages)
    if set(occupied) != set(jobs_per_record):
        raise ValueError("record/page union reconciliation failure")
    if any(n < 1 or n > RECORD // PAGE for n in occupied.values()):
        raise ValueError("record occupancy bound")
    records = len(occupied)
    # Whole-file 128KiB boundaries are assumed, NOT discovered ZFS physical blocks.
    # Last record is clipped at admitted shard EOF for the conditional read plan.
    coalesced_bytes = sum(min(RECORD, TABLE_END - record * RECORD) for record in occupied)
    return {
        "scope": "cache-empty per-chunk control; logical geometry only",
        "start_token": start, "tokens": count, "row_requests": len(rows),
        "unique_rows": len(unique_rows), "logical_requested_row_bytes": len(rows) * ROW_BYTES,
        "unique_row_payload_bytes": len(unique_rows) * ROW_BYTES,
        "row_request_page_straddlers": row_straddlers,
        "row_request_assumed_record_straddlers": record_straddlers,
        "first_page_dedup_jobs": len(jobs),
        "dedup_uses_beyond_first": len(rows) - len(jobs),
        "jobs_4k": sum(j[0] == PAGE for j in jobs.values()),
        "jobs_8k": sum(j[0] == 2 * PAGE for j in jobs.values()),
        "baseline_job_page_bytes_sum_with_overlap": baseline_bytes,
        "baseline_jobs_extending_beyond_admitted_eof": beyond_eof_jobs,
        "baseline_job_bytes_beyond_eof_geometry_only": beyond_eof_bytes,
        "distinct_expanded_pages": len(pages), "distinct_page_union_bytes": len(pages) * PAGE,
        "row_payload_assumed_records_touched": len(row_records),
        "baseline_jobs_assumed_records_touched": records,
        "record_occupancy_unique_4k_pages_histogram": histogram(occupied.values()),
        "record_baseline_job_occupancy_histogram": histogram(jobs_per_record.values()),
        "job_row_use_count_histogram": histogram(j[1] for j in jobs.values()),
        "job_unique_row_count_histogram": histogram(len(j[2]) for j in jobs.values()),
        "conditional_record_coalesced_jobs": records,
        "conditional_full_record_span_bytes": records * RECORD,
        "conditional_record_coalesced_bytes_clipped_at_shard_eof": coalesced_bytes,
        "conditional_coalesced_to_baseline_logical_byte_ratio":
            coalesced_bytes / baseline_bytes if baseline_bytes else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--fixture", required=True, choices=("32k", "32k-insert198", "256k", "tiny"))
    p.add_argument("--prefix", required=True, type=bounded_int)
    p.add_argument("--oracle-rows", required=True)
    p.add_argument("--oracle-bin", required=True, help="hashed only; never executed here")
    p.add_argument("--source-root", default=DEFAULT_ROOT)
    p.add_argument("--include-tail", action="store_true",
                   help="verify and census exactly one next token separately; oracle must include it")
    p.add_argument("--prev", nargs=2, type=bounded_int, default=(-1, -1),
                   metavar=("OLDEST", "NEWEST"), help="nondefault allowed only with tiny fixture")
    a = p.parse_args()
    if not 1 <= a.prefix <= MAX_TOKENS:
        raise ValueError("prefix outside 1..262144")
    if tuple(a.prev) != (-1, -1) and a.fixture != "tiny":
        raise ValueError("nondefault history forbidden for admitted model fixtures")
    tokens, fixture = load_tokens(checked_path(a.input), a.fixture)
    expected_tokens = a.prefix + int(a.include_tail)
    if expected_tokens > MAX_TOKENS or expected_tokens > len(tokens):
        raise ValueError("oracle prefix/tail exceeds bound or input")
    rows_path = checked_path(a.oracle_rows)
    oracle_digest = digest_file(rows_path, MAX_TOKENS * HEADS * 4)
    if oracle_digest["bytes"] != expected_tokens * HEADS * 4:
        raise ValueError("oracle row file exact size mismatch")
    root = checked_path(a.source_root)
    sources = {rel: digest_file(root / rel, 8 * 1024 * 1024) for rel in SOURCE_FILES}
    if any(sources[rel]["sha256"] != SOURCE_SHA[rel] for rel in SOURCE_FILES):
        raise ValueError("frozen source hash mismatch")
    binary = digest_file(checked_path(a.oracle_bin), 64 * 1024 * 1024)
    chunks = []
    compared = 0
    consumed_digest = hashlib.sha256()
    with rows_path.open("rb") as oracle:
        for c0 in range(0, a.prefix, CHUNK):
            n = min(CHUNK, a.prefix - c0)
            raw = oracle.read(n * HEADS * 4)
            consumed_digest.update(raw)
            if len(raw) != n * HEADS * 4:
                raise ValueError("oracle short chunk")
            rows = list(struct.unpack("<" + "I" * (n * HEADS), raw))
            for t in range(n):
                actual = rows[t * HEADS:(t + 1) * HEADS]
                expected = independent_rows(tokens, c0 + t, a.prev)
                if actual != expected:
                    h = next(h for h in range(HEADS) if actual[h] != expected[h])
                    raise ValueError("oracle mismatch token=%d head=%d actual=%d expected=%d" %
                                     (c0 + t, h, actual[h], expected[h]))
                compared += HEADS
            chunks.append(geometry(rows, c0, n))
        tail = None
        if a.include_tail:
            raw = oracle.read(HEADS * 4)
            consumed_digest.update(raw)
            if len(raw) != HEADS * 4:
                raise ValueError("oracle short tail")
            rows = list(struct.unpack("<16I", raw))
            if rows != independent_rows(tokens, a.prefix, a.prev):
                raise ValueError("oracle tail mismatch")
            compared += HEADS
            tail = geometry(rows, a.prefix, 1)
            tail["scope"] = "one-token offline tail; separate cache-empty control, no decode execution"
        if oracle.read(1):
            raise ValueError("unexpected oracle trailing bytes")
    if consumed_digest.hexdigest() != oracle_digest["sha256"]:
        raise ValueError("oracle changed between provenance and all-row verification")
    # Compact per-chunk summaries only; no whole-context row/page/record lists.
    result = {
        "success": True, "kind": "bounded CPU offline PLE census v1",
        "scope": "no actual PleReader reads/cache/performance or GPU/full-context inference qualification",
        "control": "cache empty independently at every 8192-token chunk; no 8-way replacement simulation",
        "assumed_record_layout": "131072-byte records at whole-file offset multiples; conditional only",
        "geometry": {"table_offset": TABLE_OFFSET, "row_bytes": ROW_BYTES, "table_rows": NROWS,
                     "admitted_shard_eof": TABLE_END, "page_bytes": PAGE, "assumed_record_bytes": RECORD},
        "frozen_source_commit_declared": FROZEN_COMMIT,
        "source_hashes": sources, "census_source": digest_file(Path(__file__), 1024 * 1024),
        "oracle_source": digest_file(Path(__file__).with_name("ple_ngram_oracle_v2.cpp"), 1024 * 1024),
        "oracle_binary": {"path": a.oracle_bin, **binary},
        "oracle_rows": {"path": str(rows_path), **oracle_digest},
        "fixture": fixture, "fixture_kind": a.fixture,
        "insertion": {"token": 198, "before_index": 32759} if a.fixture == "32k-insert198" else None,
        "initial_prev_oldest_first": list(a.prev),
        "pp_tokens": a.prefix, "chunk_count": len(chunks),
        "all_oracle_rows_compared": compared, "expected_rows_compared": expected_tokens * HEADS,
        "max_chunk_descriptor_uses": MAX_USES,
        "totals_sum_of_independent_chunks": {
            key: sum(c[key] for c in chunks) for key in
            ("row_requests", "unique_rows", "first_page_dedup_jobs",
             "baseline_job_page_bytes_sum_with_overlap", "distinct_page_union_bytes",
             "baseline_jobs_assumed_records_touched",
             "conditional_record_coalesced_bytes_clipped_at_shard_eof")},
        "chunks": chunks, "separate_optional_tail": tail,
    }
    if compared != expected_tokens * HEADS:
        raise ValueError("row comparison reconciliation failure")
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"success": False, "error": str(exc),
                          "no_census_aggregates_valid": True}, separators=(",", ":")), file=sys.stderr)
        sys.exit(1)
