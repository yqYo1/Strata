from pathlib import Path
import fcntl, hashlib, json
B=Path(__file__).parent
def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpp=B/'ple_ngram_oracle_v1.cpp';py=B/'ple_offline_census_v1.py'
    assert ident(cpp)['sha256']=='f5f395cb32a8948a57d522a759045619a68dd68d2dee5a3cc35feee1cac8611e'
    assert ident(py)['sha256']=='ab2d69f1d664aa96efc1994deb33777229dadb482694acd97c6ca8f4e5a96ba2'
    changes={
        'ple_ngram_oracle_v2.cpp':cpp.read_text().replace('t.reserve(limit);','t.reserve(limit+1);').replace('if(t.size()==limit) throw','if(t.size()==limit+1) throw'),
        'ple_offline_census_v2.py':py.read_text().replace('if len(tokens) >= MAX_TOKENS:', 'if len(tokens) >= MAX_TOKENS + 1:').replace('fixture == "256k" and original_count != MAX_TOKENS:', 'fixture == "256k" and original_count != MAX_TOKENS + 1:').replace('with_name("ple_ngram_oracle_v1.cpp")', 'with_name("ple_ngram_oracle_v2.cpp")')}
    for name,data in changes.items():
        q=B/name;assert not q.exists();q.write_text(data)
    proof=dict(scope='Root source-only admission/provenance correction before census execution. Pinned full fixture actually has262145 tokens; allow at most262145 originalinputIDs while keeping verified prefix/output bound262144. Original Sol sources/handoff unchanged. No tolerance, hash, numerical/geometry validation or production source changed.',originals={p.name:ident(p) for p in (cpp,py)},prepared={name:ident(B/name) for name in changes})
    q=B/'ple-offline-census-v2-source-preparation.json';assert not q.exists();q.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
