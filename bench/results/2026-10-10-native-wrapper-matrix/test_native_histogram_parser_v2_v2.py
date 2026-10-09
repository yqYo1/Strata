"""Root CPU adversarial parser qualification, no controller top-level execution."""
from pathlib import Path
import ast
import fcntl
import hashlib
import json
import os
import re
import time

B = Path(__file__).parent
SOURCE = B / 'run_owned_native_dispatch_histogram_v0141_code32k_v2.py'
SOURCE_PIN = 'ff6d8ad537b9e9fb81511725dd5cf936a3a0d460bae271d79707777f38d0a599'
PRODUCER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-dispatch-histogram-v0141-20261009/src/kernels/cpu/pool.cpp')
OUT = B / 'native-histogram-parser-v2-cpu-validation-v2.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE) == SOURCE_PIN
assert sha(PRODUCER) == '5ef33f7d2020a17ecc36ad15796c6f021b4b5963998aa964dccedbe733637368'
assert not OUT.exists()
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
started = time.monotonic()
# Extract only parser constants/function; importing the full owner would execute
# admission/model branches. No source rewriting or fake device features occur.
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
names = {'HISTOGRAM_PREFIX', 'HISTOGRAM_LEGEND', 'HISTOGRAM_END', 'HISTOGRAM_KEYS'}
nodes = [node for node in tree.body if
         isinstance(node, ast.FunctionDef) and node.name == 'parse_histogram' or
         isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
assert len(nodes) == 5
namespace = {'re': re}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
parse = namespace['parse_histogram']
# Independently transcribed frozen producer fprintf schema and legal counter
# examples. Producer source pin binds the layout; synthetic counts are not model.
header = ('strata native dispatch exposure: decode request delta; cpu2=1 cpu512=0 hybrid=0 '
          'gu512=0 gu256=1 gu_kq=0 down_kq=0 iq4nl=1 gu_min=2 iq3s_min=2 down_min=2 '
          'gather_setting=-1 iq256_variant_capabilities=1; '
          'eligible bits=IQ512:1,IQ256:2,KQ256:4,IQ4NL:8,Q2:16; '
          'variant=scalar:0,gather:1,vnni:2,both:3,worker-unresolved:-1; '
          'cells=phase/type/nt/selected/eligible/variant:experts,tokens,output_rows')
tail = '; exposure only, not time/traffic/worker imbalance'
cells = ['GU/18/1/ggml/2/0:2,2,1280', 'GU/22/2/iq256/2/0:3,6,1920',
         'Down/20/1/ggml/8/0:2,2,5120', 'Down/42/2/q2-avx2/16/0:3,6,7680']
valid = header + ' ' + ' '.join(cells) + tail
all_types = [f'GU/{t}/{n}/{"ggml" if n == 1 else "iq256"}/2/0:1,{n},640'
             for t in [18,21,22,23] for n in [1,2]]
all_types += [f'Down/{t}/{n}/{("ggml" if n == 1 else "iq4nl256") if t == 20 else "q2-avx2"}/{8 if t == 20 else 16}/0:2,{2*n},5120'
              for t in [20,42] for n in [1,2]]
tests = [('valid-mixed', valid, True), ('valid-all-types', header+' '+' '.join(all_types)+tail, True),
         ('duplicate-cell', header+' '+' '.join(cells+[cells[0]])+tail, False),
         ('empty', header+tail, False), ('missing-down', header+' '+' '.join(cells[:2])+tail, False),
         ('truncated-tail', valid[:-1], False), ('trailing-garbage', valid+' x', False),
         ('wrong-cpu', valid.replace('cpu2=1','cpu2=0'), False),
         ('wrong-min', valid.replace('gu_min=2','gu_min=1'), False),
         ('header-duplicate', valid.replace('cpu512=0','cpu2=1'), False),
         ('unknown-type', valid.replace('GU/18/', 'GU/42/'), False),
         ('unknown-label', valid.replace('/ggml/', '/unknown/', 1), False),
         ('wrong-selected', valid.replace('GU/18/1/ggml/', 'GU/18/1/iq256/'), False),
         ('wrong-eligible', valid.replace('GU/18/1/ggml/2/', 'GU/18/1/ggml/0/'), False),
         ('wrong-variant', valid.replace('GU/18/1/ggml/2/0:', 'GU/18/1/ggml/2/1:'), False),
         ('bad-tokens', valid.replace(':2,2,1280', ':2,3,1280'), False),
         ('bad-rows', valid.replace(':2,2,1280', ':2,2,1279'), False),
         ('zero-count', valid.replace(':2,2,1280', ':0,0,0'), False),
         ('overflow', valid.replace(':2,2,1280', ':18446744073709551616,18446744073709551616,11805916207174113034240'), False),
         ('counts-disagree', valid.replace('Down/20/1/ggml/8/0:2,2,5120','Down/20/1/ggml/8/0:1,1,2560'), False),
         ('nt-zero', valid.replace('GU/18/1/', 'GU/18/0/'), False),
         ('nt3-falsifies-config', valid.replace('GU/22/2/iq256/2/0:3,6,1920','GU/22/3/iq256/2/0:3,9,1920'), False),
         ('nan-count', valid.replace(':2,2,1280', ':nan,2,1280'), False)]
results = []
for name, text, wanted in tests:
    assert time.monotonic() - started < 60
    try:
        parsed = parse(text)
        accepted = True
        error = None
        if wanted:
            assert parsed['totals']['GU']['experts'] == (5 if name == 'valid-mixed' else 8)
            assert parsed['totals']['GU']['tokens'] == (8 if name == 'valid-mixed' else 12)
            assert parsed['source_bound_max_nt'] == 2
    except (AssertionError, ValueError) as exc:
        accepted, parsed, error = False, None, repr(exc)
    results.append({'name':name,'expected_accept':wanted,'accepted':accepted,'passed':accepted==wanted,
                    'fixture_sha256':hashlib.sha256(text.encode()).hexdigest(),'error':error,
                    'parsed':parsed})
record = {'active':False,'complete':True,'passed':all(r['passed'] for r in results),
          'process_identity':{'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])},
          'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'deadline_seconds':60,
          'gpu_executed':False,'model_opened':False,'runtime_controller_executed':False,
          'source_sha256':SOURCE_PIN,'test_sha256':sha(__file__),'producer_sha256':sha(PRODUCER),
          'cases':results,'case_count':len(results),'elapsed_seconds':time.monotonic()-started,
          'adopted':False,'performance_eligible':False}
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed':record['passed'],'cases':len(results),'failed':[r['name'] for r in results if not r['passed']]}))
if not record['passed']:
    raise SystemExit(1)
