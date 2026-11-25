"""Check the actual batch path and preserve the matched prefill/decode work."""

import fcntl
import json
import re
import sys

sys.dont_write_bytecode = True
import dflash2_primary_prefill_curve as curve

bank, require, footer = curve.bank, curve.require, curve.speed.clock.footer
READ = 102401
CONTROL = '0577a8bbca081ba5ec5d03d32d3ed4376d7d553f'
LOG_SHA256 = 'aa61b30d10ea6abfc1b0bc264c6e0aa00d0b2b743d969699e6d68d984e5a9d6e'


def route(frame, plan):
    arm = plan['arm']
    require(arm in ('demand', 'batched') and plan['read'] == READ and plan['kv_arm'] == 'retain',
            'unexpected upload mode, input extent or KV policy')
    info = curve.route(frame, plan, batched_upload=arm == 'batched')
    counts, = re.findall(footer.COPY, frame, re.M)
    decode = dict(zip(footer.COPY_KEYS, map(int, counts), strict=True))
    require(all(decode[name] == 0 for name in footer.COPY_KEYS[:6])
            and 0 < decode['cpu_unique_misses'] <= decode['cpu_entries'],
            'the decode expert placement or CPU fallback changed')
    require(info['native_request'][:4] == [READ, 0, READ, 24],
            'the actual request differs from the timed workload')
    return dict(info, decode_experts=decode)


def comparable_work(info):
    work = {key: value for key, value in info.items() if key != 'free_bytes'}
    work['uploads'] = {key: value for key, value in info['uploads'].items() if key != 'permanent_slots'}
    return work


def replay():
    source = curve.ROOT / '3' / 'engine.stderr'
    history = bank.previous.historic(CONTROL, curve.KEY)
    control = history['processes'][3]
    require(control['read'] == READ and control['passed'] and control['complete']
            and control['ownership_closed'] and control['native_exit_code'] == 0
            and not control['fresh_faults'] and bank.sha(source) == LOG_SHA256,
            'the closed demand control or exact native log changed')
    text = source.read_text()
    starts = [m.start() for m in re.finditer(r'^strata prefill heads: draft_reader=absent-dflash2$', text, re.M)]
    require(len(starts) == len(control['routes']) == 3, 'recover all three original request frames')
    demand = dict(read=READ, arm='demand', kv_arm='retain')
    batched = dict(demand, arm='batched')
    checked = 0
    for start, end, expected in zip(starts, starts[1:] + [len(text)], control['routes'], strict=True):
        frame = text[start:end]
        observed = route(frame, demand)
        require({k: v for k, v in observed.items() if k != 'decode_experts'} == expected,
                'demand route differs from the original committed receipt')
        count = expected['uploads']['uploads']
        footer_line, = [line for line in frame.splitlines()
                        if line.startswith('strata MoE expert bank complete: ')]
        marker = f'strata MoE split expert upload: windows=48 experts={count} copy_commands={count} final_drain=checked'
        synthetic = frame.replace(footer_line, footer_line.replace('permanent_slots=31', 'permanent_slots=0')
                                  + '\n' + marker, 1)
        candidate = route(synthetic, batched)
        require(comparable_work(candidate) == comparable_work(observed),
                'another counter changed between the demand and synthetic batch routes')
        invalid = (
            (frame, batched),  # silent demand fallback
            (synthetic, demand),
            (synthetic.replace(marker, marker.replace('windows=48', 'windows=47')), batched),
            (synthetic.replace(marker, marker.replace(f'copy_commands={count}', 'copy_commands=1')), batched),
            (synthetic.replace('submitted_blobs=0', 'submitted_blobs=1'), batched),
        )
        for altered, plan in invalid:
            try:
                route(altered, plan)
            except RuntimeError:
                checked += 1
            else:
                raise RuntimeError('the route reader accepted a changed upload or decode path')
    return dict(archived_demand_requests=len(starts), synthetic_batched_requests=len(starts),
                expected_refusals=checked, gpu_executed=False)


if __name__ == '__main__':
    require(sys.argv[1:] == ['replay'], 'use replay for the bounded host check')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(replay(), sort_keys=True))
