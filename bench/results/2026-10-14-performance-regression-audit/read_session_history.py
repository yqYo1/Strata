#!/usr/bin/env python3
"""Extract user-visible performance claims from this exact thread, not tool/env logs."""
import hashlib
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
D = Path(__file__).parent
LOG = Path('/home/yayoi/.codex/sessions/2026/10/02/rollout-2026-10-02T19-12-34-01a0fc19-f45e-7c80-be7c-d1e00e327d4b.jsonl')
PERF = re.compile(r'(?:\d[\d.,]*\s*(?:tok(?:en)?s?/s|t/s|tps)|\b(?:prefill|decode|PP|TG)\b.*\d)', re.I)
SECRET = re.compile(r'(?i)(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|authorization\s*:|bearer\s+|(?:api_key|api-key)\s*[=:])')


def main():
    size = LOG.stat().st_size
    offset = 0
    digest = hashlib.sha256()
    claims = []
    first_xe = None
    meta = None
    with LOG.open('rb') as stream:
        for line_number, raw in enumerate(stream, 1):
            if offset + len(raw) > size:
                break
            start = offset
            offset += len(raw)
            digest.update(raw)
            row = json.loads(raw)
            payload = row.get('payload', {})
            if row.get('type') == 'session_meta':
                meta = {k:payload.get(k) for k in ('id', 'timestamp', 'cwd')}
                assert meta['id'] == '01a0fc19-f45e-7c80-be7c-d1e00e327d4b'
            if row.get('type') != 'response_item' or payload.get('type') != 'message':
                continue
            text = '\n'.join(x.get('text', '') for x in payload.get('content', [])
                if x.get('type') in ('input_text', 'output_text'))
            role = payload.get('role')
            if role == 'user' and 'github.com/MistVVK/XeStrata' in text and first_xe is None:
                first_xe = dict(timestamp=row['timestamp'], line=line_number, byte_offset=start,
                                text=text)
            phase = payload.get('channel') or payload.get('phase')
            if role != 'assistant' or phase not in ('final', 'final_answer', 'commentary'):
                continue
            selected = [x for x in text.splitlines() if PERF.search(x) and not SECRET.search(x)]
            if not selected:
                continue
            # Only user-visible metric lines, no tool calls/outputs, env/config or hidden reasoning.
            claims.append(dict(timestamp=row['timestamp'], line=line_number, byte_offset=start,
                channel=phase, before_first_Xe_request=first_xe is None,
                message_sha256=hashlib.sha256(text.encode()).hexdigest(), metric_lines=selected))
    assert meta and first_xe
    result = dict(session=meta, original_path=str(LOG), snapshot_bytes=size,
        scanned_complete_line_bytes=offset, scanned_prefix_sha256=digest.hexdigest(),
        first_Xe_user_request=first_xe, claims=claims,
        scope='user-visible historical performance claims; must be tied to original measurements before treating numbers as verified performance')
    (D/'session-performance-claims.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    before = [x for x in claims if x['before_first_Xe_request']]
    print(json.dumps(dict(before_Xe_claim_messages=len(before), after_Xe_claim_messages=len(claims)-len(before),
        first_Xe_request=first_xe, last_before_Xe=before[-20:]), indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
