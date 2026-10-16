#!/usr/bin/env python3
"""Recheck pinned, user-visible historical messages without scanning private payloads."""
import hashlib
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
D = Path(__file__).parent
PRIOR = P / 'bench/results/2026-10-14-performance-regression-audit'
SECRET = re.compile(r'(?i)(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|authorization\s*:|bearer\s+|(?:api_key|api-key)\s*[=:])')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    claims_raw = (PRIOR / 'session-performance-claims.json').read_bytes()
    decisions_raw = (PRIOR / 'session-decisions.json').read_bytes()
    claims = json.loads(claims_raw)
    decisions = json.loads(decisions_raw)
    source = Path(claims['original_path'])
    assert source.name == 'rollout-2026-10-02T19-12-34-01a0fc19-f45e-7c80-be7c-d1e00e327d4b.jsonl'
    wanted = {}
    for entry in claims['claims'] + decisions['messages']:
        offset = entry['byte_offset']
        if offset in wanted:
            assert wanted[offset]['message_sha256'] == entry['message_sha256']
        wanted[offset] = entry
    checks = []
    selected = []
    retained_dates = {'2026-10-05', '2026-10-08', '2026-10-09', '2026-10-10'}
    with source.open('rb') as handle:
        for offset, entry in sorted(wanted.items()):
            handle.seek(offset)
            row = json.loads(handle.readline())
            payload = row.get('payload', {})
            assert row.get('type') == 'response_item' and payload.get('type') == 'message'
            role = payload.get('role')
            phase = payload.get('channel') or payload.get('phase')
            assert role == 'user' or (role == 'assistant' and phase in ['final', 'final_answer', 'commentary'])
            # Only inspect content after its role/channel has passed the public-message gate.
            text = '\n'.join(c.get('text', '') for c in payload.get('content', [])
                             if c.get('type') in ['input_text', 'output_text'])
            assert row['timestamp'] == entry['timestamp']
            assert sha(text.encode()) == entry['message_sha256']
            assert not SECRET.search(text)
            if 'metric_lines' in entry:
                assert all(line in text.splitlines() for line in entry['metric_lines'])
            if 'text' in entry:
                assert text == entry['text']
            check = dict(byte_offset=offset, timestamp=row['timestamp'], role=role,
                         phase=phase, message_sha256=entry['message_sha256'])
            checks.append(check)
            if entry in decisions['messages'] and row['timestamp'][:10] in retained_dates:
                selected.append(dict(check, evidence_reference='session-decisions.json',
                                     public_text=text))
    result = dict(passed=True, original_path=str(source), current_source_bytes=source.stat().st_size,
                  original_snapshot_bytes=claims['snapshot_bytes'],
                  original_snapshot_prefix_sha256=claims['scanned_prefix_sha256'],
                  original_snapshot_digest_recomputed=False,
                  index_sha256={'claims': sha(claims_raw), 'decisions': sha(decisions_raw)},
                  original_claim_count=len(claims['claims']), original_decision_count=len(decisions['messages']),
                  verified_unique_public_messages=len(checks),
                  verified_public_message_checks_sha256=sha(json.dumps(checks, sort_keys=True).encode()),
                  selected_public_message_checks=[{k: v for k, v in x.items()
                                                  if k not in ['public_text', 'evidence_reference']}
                                                 for x in selected],
                  selected_decision_offsets=[x['byte_offset'] for x in selected],
                  scope='Direct seek and original-text hash checks of pinned public messages only. No tool/analysis payloads, environment, full log copy or new model run. Original whole-prefix digest is historical metadata, not recomputed proof.',
                  decisions='PR6 453->646 was reported before the first Xe request; original 32K paired receipts, source route and later full-context/MTP results remain separate evidence. No current speed/correctness/quality qualification.')
    # Full selected text is already committed in the canonical decisions index; do not duplicate it.
    (D / 'public-history-reverification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=True, verified_unique_public_messages=len(checks),
                          selected_decision_offsets=result['selected_decision_offsets']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
