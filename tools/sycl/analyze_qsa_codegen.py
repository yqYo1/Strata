#!/usr/bin/env python3
"""Offline, conservative loop attribution for the pinned QSA dump capture.

Counts are static instruction annotations, never dynamic traffic or latency.
LLVM/vISA IDs can occur in several blocks; preserve ambiguous/unmapped cases.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

P = Path(__file__).resolve().parents[2]
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-codegen-current.json'
CAPTURE_COMMIT = '6ab472964'
LABEL = r'[-\w.$]+'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], timeout=20)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def natural_loops(edges, entry):
    predecessors = collections.defaultdict(set)
    for block, targets in edges.items():
        require(targets <= edges.keys(), 'unresolved CFG target')
        for target in targets:
            predecessors[target].add(block)
    reached, todo = set(), [entry]
    while todo:
        block = todo.pop()
        if block not in reached:
            reached.add(block)
            todo.extend(edges[block])
    dominators = {b: ({b} if b == entry else set(reached)) for b in reached}
    changed = True
    while changed:
        changed = False
        for block in reached - {entry}:
            incoming = predecessors[block] & reached
            require(incoming, 'reachable block without predecessor')
            new = {block} | set.intersection(*(dominators[p] for p in incoming))
            if new != dominators[block]:
                dominators[block] = new
                changed = True
    loops = collections.defaultdict(set)
    for latch in reached:
        for header in edges[latch]:
            if header not in dominators[latch]:
                continue
            members, todo = {header}, [latch]
            while todo:
                block = todo.pop()
                if block not in members:
                    members.add(block)
                    todo.extend(predecessors[block] & reached)
            loops[header].update(members)
    return reached, loops, predecessors


def llvm_cfg(text):
    blocks = {'<entry>': []}
    current = '<entry>'
    started = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith('define spir_kernel '):
            require(not started, 'multiple kernel functions')
            started = True
            continue
        if not started:
            continue
        if line == '}':
            break
        label = re.match(r'^(' + LABEL + r'):', line)
        if label:
            current = label[1]
            require(current not in blocks, 'duplicate LLVM block')
            blocks[current] = []
        elif line.strip() and not line.lstrip().startswith(';'):
            blocks[current].append((number, line))
    require(started, 'missing LLVM kernel')
    edges = {}
    for block, lines in blocks.items():
        terminators = [line.split(';', 1)[0].strip() for _, line in lines
                       if re.match(r'\s*(?:br|ret)\s', line)]
        require(len(terminators) == 1, 'expected one br/ret in ' + block)
        targets = set(re.findall(r'label %(' + LABEL + r')', terminators[0]))
        edges[block] = targets
    reached, loops, _ = natural_loops(edges, '<entry>')
    return blocks, reached, loops


def selected_loop(blocks, loops):
    # The zero-based i64 loop incremented by Cells=32, not an inner stride32
    # address expression. Require its four FP32 PV recurrence fragments too.
    candidates = []
    additions = {}
    for block, lines in blocks.items():
        for number, line in lines:
            match = re.search(r'(%' + LABEL + r') = add (?:nuw |nsw )*i64 (%' + LABEL + r'), 32(?:,|\s)', line)
            if match:
                additions[match[1]] = (match[2], block, number, int(re.search(r'; visa id: (\d+)', line)[1]))
    for header, members in loops.items():
        for number, line in blocks[header]:
            match = re.search(r'(%' + LABEL + r') = phi i64 \[ 0, %(' + LABEL + r') \], \[ (%' + LABEL + r'), %(' + LABEL + r') \]', line)
            if not match or match[3] not in additions:
                continue
            base, update_block, update_line, update_id = additions[match[3]]
            if base != match[1] or update_block not in members or match[4] not in members or match[2] in members:
                continue
            accumulators = sum(' = phi <8 x float> ' in value for _, value in blocks[header])
            if accumulators == 4:
                candidates.append(dict(header=header, members=members, phi=match[1],
                    phi_line=number, update_line=update_line, update_id=update_id, step=32, fp32_accumulator_phis=accumulators))
    require(len(candidates) == 1, 'selected-cell loop is not uniquely qualified')
    return candidates[0]


def scratch_kind(line):
    return 'spill' if 'spill to offset[' in line else 'fill' if 'fill from offset[' in line else None


def native_attribution(asm, update_id):
    blocks, edges, declared_preds = {}, {}, {}
    current = None
    for number, line in enumerate(asm.splitlines(), 1):
        match = re.match(r'// (B\d+):.*Preds:\{([^}]*)\},\s+Succs:\{([^}]*)\}', line)
        if match:
            current = match[1]
            require(current not in blocks, 'duplicate native block')
            blocks[current] = []
            declared_preds[current] = set(re.findall(r'B\d+', match[2]))
            edges[current] = set(re.findall(r'B\d+', match[3]))
        elif current is not None:
            blocks[current].append((number, line))
    require('B000' in blocks, 'missing native entry')
    reached, loops, preds = natural_loops(edges, 'B000')
    require(all(declared_preds[b] == preds[b] for b in blocks), 'native predecessor/successor mismatch')
    updates = []
    for block, lines in blocks.items():
        for number, line in lines:
            if re.search(r'\badd\s.*32:w\s.*; \$' + str(update_id) + r'$', line):
                updates.append((block, number))
    require(len(updates) == 1, 'native selected-cell increment not unique')
    update_block, update_line = updates[0]
    candidates = [(h, m) for h, m in loops.items() if update_block in m and h in edges[update_block]]
    require(len(candidates) == 1, 'native outer-loop backedge not unique')
    header, members = candidates[0]
    counts = {kind: collections.Counter() for kind in ('spill', 'fill')}
    for block, lines in blocks.items():
        for _, line in lines:
            kind = scratch_kind(line)
            if kind:
                counts[kind]['inside' if block in members else 'outside' if block in reached else 'unreachable'] += 1
    scratch_sends = [line for line in asm.splitlines() if 'ss[a0.2]' in line and not line.startswith('//')]
    require(all(scratch_kind(line) for line in scratch_sends), 'unannotated scratch-surface send')
    require(sum(sum(c.values()) for c in counts.values()) == len(scratch_sends), 'scratch-send conservation failed')
    return dict(blocks=len(blocks), reachable_blocks=len(reached), loop_header=header,
        loop_blocks=len(members), update_block=update_block, update_line=update_line,
        annotated_scratch_instructions={k: dict(total=sum(v.values()), regions=dict(sorted(v.items()))) for k, v in counts.items()})


def analyze(asm, llvm):
    blocks, reached, loops = llvm_cfg(llvm)
    loop = selected_loop(blocks, loops)
    ids = collections.defaultdict(set)
    for block, lines in blocks.items():
        for _, line in lines:
            match = re.search(r'; visa id: (\d+)', line)
            if match:
                ids[int(match[1])].add(block)
    counts = {kind: collections.Counter() for kind in ('spill', 'fill')}
    for line in asm.splitlines():
        kind = scratch_kind(line)
        if kind is None:
            continue
        match = re.search(r';\s*\$(\d+)\s*$', line)
        mapped = ids.get(int(match[1]), set()) if match else set()
        inside = mapped & loop['members']
        region = ('unmapped' if not mapped else 'unreachable' if not mapped <= reached
                  else 'inside' if inside == mapped else 'outside' if not inside else 'ambiguous')
        counts[kind][region] += 1
    spilled = [line for line in asm.splitlines() if line.startswith('//.declare ') and ' rf=r ' in line and '(spilled' in line]
    types = collections.Counter(re.search(r'\btype=(\w+)', line)[1] for line in spilled)
    headers = {}
    for key in ('instCount', 'spill size', 'spill GRF est. ref count', 'spill flag store', 'spill flag load'):
        match = re.search(r'^//\.' + re.escape(key) + r'\s+(\d+)\s*$', asm, re.M)
        require(match, 'missing IGC header: ' + key)
        headers[key] = int(match[1])
    return dict(compiler_headers=headers, spilled_declarations=len(spilled), declaration_types=dict(sorted(types.items())),
        llvm_blocks=len(blocks), reachable_blocks=len(reached), natural_loop_headers=len(loops),
        selected_loop={k: v for k, v in loop.items() if k != 'members'},
        selected_loop_blocks=len(loop['members']),
        llvm_origin_scratch_instructions={k: dict(total=sum(v.values()), regions=dict(sorted(v.items()))) for k, v in counts.items()},
        native_cfg=native_attribution(asm, loop['update_id']))


def main():
    reached, loops, _ = natural_loops(dict(entry={'head'}, head={'body', 'exit'},
        body={'head'}, exit=set(), unreachable={'unreachable'}), 'entry')
    require(reached == {'entry', 'head', 'body', 'exit'} and dict(loops) == {'head': {'head', 'body'}},
        'natural-loop/unreachable-block self-check failed')
    relative = str(Path(__file__).resolve().relative_to(P))
    revision = git('rev-parse', 'HEAD').decode().strip()
    require(git('show', revision + ':' + relative) == Path(__file__).read_bytes(), 'commit analyzer before execution')
    current = json.loads(RESULT.read_text())
    require(git('show', revision + ':' + str(RESULT.relative_to(P))) == RESULT.read_bytes(), 'commit analysis plan first')
    original = json.loads(git('show', CAPTURE_COMMIT + ':' + str(RESULT.relative_to(P))))
    require(current['checks'] == original['checks'], 'do not alter original capture evidence')
    check = original['checks'][-1]
    require(check['passed'] and check['complete'] and check['ownership_closed'] and not check['active'] and check['fresh_faults'] == [], 'capture not qualified')
    require(current['static_analysis_plan']['capture_commit'] == CAPTURE_COMMIT and current['static_analysis_plan']['version'] == 2, 'unexpected analysis plan')
    capture = check['capture']
    files = {}
    for entry in capture['selected_files']:
        path = Path(entry['path'])
        data = path.read_bytes()
        require(len(data) == entry['bytes'] and digest(data) == entry['sha256'], 'capture input changed: ' + str(path))
        files[str(path)] = data
    arms = []
    for number, item in enumerate(capture['executed']):
        path = Path(item['asm']['path'])
        asm = files[str(path)].decode()
        require(asm.splitlines()[0] == '//.kernel ' + item['kernel'], 'executed name mismatch')
        llvm = files[str(path.with_suffix('.visa.ll'))].decode()
        require('@' + item['kernel'] + '(' in llvm, 'LLVM name mismatch')
        arms.append(dict(arm=number, asm_sha256=item['asm']['sha256'], **analyze(asm, llvm)))
    require(len(arms) == 3 and arms[0]['compiler_headers'] == arms[1]['compiler_headers'], 'namespace control mismatch')
    dat = [files[str(Path(e['asm']['path']).with_suffix('.dat'))] for e in capture['executed']]
    current['static_analysis'] = dict(analyzer_commit=revision, analyzer_sha256=digest(Path(__file__).read_bytes()),
        arms=arms, default_control_dat_identical=dat[0] == dat[1],
        limits='Native CFG membership uses compiler-emitted BNNN predecessor/successor metadata and the selected-loop i64+32 backedge; LLVM-origin counts are separate. Neither implies every iteration/lane executes a static instruction. Shared visa IDs are not unique source-value provenance. rf=r spilled declarations include aliases; declaration types do not identify named tensors. Compiler estimates/flag counters are not observed traffic. No occupancy, causal attribution, speed or loaded-machine-code-equivalence claim.')
    RESULT.write_text(json.dumps(current, indent=2) + '\n')
    print(json.dumps(dict(analysis_complete=True, arms=arms, default_control_dat_identical=dat[0] == dat[1])))


if __name__ == '__main__':
    main()
