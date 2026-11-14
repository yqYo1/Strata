#!/usr/bin/env python3
"""Inspect the linked C4 DPAS image for the upcoming decode-head integration.

CPU controls: python -B -m unittest -v tools.sycl.test_q6_dpas_component
The closed native comparison and its execution recipe are recoverable from
f97893576704e6679c89ea6c4613d912665a0824. Retain only the live image reader here.
Main-authored;repository LICENSE.
"""
import hashlib
import struct
import sys
sys.dont_write_bytecode = True
from measurement_guards import require

KERNEL_ENTRY = '_ZTSN6strata7kernels9component12_GLOBAL__N_18Q6DpasC4E'
def embedded_dpas(blob, *, allow_shared_image=False):
    """Read exact linked ELF images and count reachable SPIR-V DPAS calls.

    This is a bounded format/callgraph check,not a full SPIR-V validator or
    evidence of final native ISA,hardware utilization or dynamic call count.
    """
    require(64 <= len(blob) <= 64 << 20, 'finite ELF extent')
    def span(offset, size):
        require(0 <= offset <= len(blob) and 0 <= size <= len(blob)-offset,
                'ELF file span')
        return blob[offset:offset+size]
    def name(table, at):
        require(0 <= at < len(table), 'ELF/SPIR-V string offset')
        end=table.find(b'\0',at)
        require(at <= end <= at+4096, 'bounded terminated name')
        return table[at:end].decode('utf-8')
    h=struct.unpack('<16sHHIQQQIHHHHHH',span(0,64))
    require(h[0][:6]==b'\x7fELF\x02\x01' and h[1] in (2,3) and h[2]==62
        and h[8]==64 and h[11]==64 and 0<h[12]<=4096 and h[13]<h[12],
        'supported ELF64 little-endian native executable')
    sections=[struct.unpack('<IIQQQQIIQQ',span(h[6]+64*i,64)) for i in range(h[12])]
    strings=sections[h[13]]
    require(strings[1]==3, 'ELF section-name string table')
    section_names=span(strings[4],strings[5])
    ids={i for i,q in enumerate(sections)
        if name(section_names,q[0]).startswith('__CLANG_OFFLOAD_BUNDLE__sycl-spir64')}
    require(ids, 'linked SYCL SPIR-V section')
    seen=set();matches=[]
    for table in sections:
        if table[1]!=2:continue
        require(table[9]==24 and table[5]%24==0 and table[5]//24<=1000000
            and table[6]<len(sections), 'finite ELF symbol table')
        string_section=sections[table[6]]
        require(string_section[1]==3, 'ELF symbol-name string table')
        symbol_names=span(string_section[4],string_section[5])
        for at in range(table[4],table[4]+table[5],24):
            symbol,_,_,section,value,size=struct.unpack('<IBBHQQ',span(at,24))
            if section not in ids or size==0:continue
            q=sections[section]
            require(q[3]<=value and size<=q[5] and value-q[3]<=q[5]-size,
                    'embedded image belongs to its ELF section')
            offset=q[4]+value-q[3];key=(offset,size)
            image=span(offset,size)
            if key in seen or image[:4]!=b'\x03\x02\x23\x07':continue
            seen.add(key)
            require(len(seen)<=1024, 'finite linked image inventory')
            if KERNEL_ENTRY.encode()+b'\0' not in image:continue
            require(20<=size and size%4==0, 'SPIR-V module extent')
            words=struct.unpack('<'+str(size//4)+'I',image)
            require(words[0]==0x07230203 and 0x10000<=words[1]<=0x10600
                and 0<words[3]<=10000000 and words[4]==0, 'SPIR-V module header')
            labels={};entries=[];functions=set();calls=[];function=None
            cursor=5;instructions=0
            while cursor<len(words):
                count,op=words[cursor]>>16,words[cursor]&65535
                require(count>0 and count<=len(words)-cursor, 'SPIR-V instruction extent')
                a=words[cursor+1:cursor+count];instructions+=1
                require(instructions<=1000000, 'finite SPIR-V instruction inventory')
                if op==5:
                    require(len(a)>=2 and 0<a[0]<words[3], 'SPIR-V OpName')
                    text=name(struct.pack('<'+str(len(a)-1)+'I',*a[1:]),0)
                    require(a[0] not in labels or labels[a[0]]==text, 'unambiguous SPIR-V names')
                    labels[a[0]]=text
                elif op==15:
                    require(len(a)>=3 and 0<a[1]<words[3], 'SPIR-V OpEntryPoint')
                    text=name(struct.pack('<'+str(len(a)-2)+'I',*a[2:]),0)
                    entries.append((a[0],a[1],text))
                elif op==54:
                    require(len(a)==4 and function is None and 0<a[1]<words[3]
                        and a[1] not in functions, 'SPIR-V function declaration')
                    function=a[1];functions.add(function)
                elif op==56:
                    require(not a and function is not None, 'SPIR-V function end')
                    function=None
                elif op==57:
                    require(len(a)>=3 and function is not None and 0<a[2]<words[3],
                            'SPIR-V function call')
                    calls.append((function,a[2]))
                cursor+=count
            require(function is None, 'closed SPIR-V function')
            chosen=[entry for entry in entries if entry[2]==KERNEL_ENTRY]
            if not chosen:continue
            require(len(chosen)==1, 'one exact named C4 kernel entrypoint')
            if allow_shared_image:
                require(len(entries)<=256 and len({v[1] for v in entries})==len(entries)
                    and len({v[2] for v in entries})==len(entries)
                    and all(v[0]==6 and v[1] in functions for v in entries),
                    'distinct defined kernel entrypoints in shared image')
            else:
                require(len(entries)==1 and entries[0][0]==6 and entries[0][1] in functions,
                        'one exact C4 kernel entrypoint')
            reachable=set();pending=[chosen[0][1]];edges={}
            for caller,callee in calls:edges.setdefault(caller,[]).append(callee)
            while pending:
                f=pending.pop()
                if f in reachable:continue
                reachable.add(f);pending.extend(edges.get(f,()))
            dpas=[(caller,callee) for caller,callee in calls if caller in reachable and
                labels.get(callee,'').startswith(('llvm.genx.dpas.','llvm.genx.dpas2.'))]
            require(len(dpas)>=2 and all(callee in functions for _,callee in dpas),
                    'at least two reachable SPIR-V DPAS calls')
            matches.append(dict(image_symbol=name(symbol_names,symbol),image_offset=offset,
                image_bytes=size,image_sha256=hashlib.sha256(image).hexdigest(),
                entrypoint=KERNEL_ENTRY,reachable_functions=len(reachable),
                dpas_static_calls=len(dpas),dpas_callees=sorted({labels[v] for _,v in dpas})))
            if allow_shared_image:
                matches[-1]['module_kernel_entrypoints']=[v[2] for v in entries]
    require(len(matches)==1, 'one exact linked C4 device image')
    return dict(matches[0],device_images=len(seen),
        scope='Reachable static GenX DPAS calls in exact linked ELF SPIR-V;not final native ISA,utilization or dynamic call count.')


def spirv_structure(raw):
    """Compare our finite C4 modules without depending on linker ID ordering.

    Own code against Khronos SPIRV-Headers core grammar (1.6 revision7), SHA256
    8f20f6ca4c2411391d14125cac69999f1c8248ef656e00d9d534e5e2b7a4f52c:
    https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.core.grammar.json
    Preserve every instruction, operand, annotation and function-body order.
    Only IDs and global/function declaration order are canonicalized. Reject
    unreviewed opcodes/decorations/modes and ambiguous identities. Not an ISA,
    general SPIR-V validity, runtime safety or performance certificate.
    """
    require(20 <= len(raw) <= 1 << 20 and len(raw) % 4 == 0, 'bounded SPIR-V module')
    w = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
    require(w[0] == 0x07230203 and w[4] == 0 and 0 < w[3] <= 100000, 'SPIR-V header')
    operations = []; at = 5
    while at < len(w):
        count, op = w[at] >> 16, w[at] & 65535
        require(count > 0 and at + count <= len(w), 'SPIR-V instruction extent')
        operations.append((op, tuple(w[at + 1:at + count]))); at += count
    def literal_string(a, start):
        for end in range(start + 1, len(a) + 1):
            b = struct.pack('<' + 'I' * (end - start), *a[start:end])
            if b'\0' in b:
                split = b.index(b'\0')
                require(not any(b[split:]), 'SPIR-V string padding')
                return b[:split].decode('utf-8'), end
        raise RuntimeError('unterminated SPIR-V string')
    typed = {1, 41, 43, 44, 46, 55, 57, 61, 67, 70, 79, 81, 82,
             111, 113, 114, 115, 117, 121, 124, 128, 129, 130, 132, 133, 135,
             169, 170, 176, 177, 194, 196, 197, 198, 199, 245}
    fixed = {1: 2, 3: 2, 14: 2, 17: 1, 19: 1, 20: 1, 21: 3, 22: 2, 23: 3,
             32: 3, 41: 2, 46: 2, 54: 4, 55: 2, 56: 0, 111: 3, 113: 3,
             114: 3, 115: 3, 117: 3, 121: 3, 124: 3, 128: 4, 129: 4,
             130: 4, 132: 4, 133: 4, 135: 4, 169: 5, 170: 4, 176: 4,
             177: 4, 194: 4, 196: 4, 197: 4, 198: 4, 199: 4, 246: 3,
             248: 1, 249: 1, 253: 0}
    def schema(op, a):
        if op in fixed: require(len(a) == fixed[op], 'SPIR-V fixed operand count')
        if op in (3, 14, 17, 56, 253): ids = set()
        elif op in (5, 10, 11):
            _, end = literal_string(a, int(op != 10)); require(end == len(a), 'SPIR-V string extent')
            ids = {0} if op != 10 else set()
        elif op == 15:
            require(len(a) >= 3, 'SPIR-V entry operands')
            _, end = literal_string(a, 2); ids = {1, *range(end, len(a))}
        elif op == 16:
            require(len(a) >= 2 and a[1] in (31, 35, 5618, 6417) and
                len(a) == (2 if a[1] == 31 else 3), 'unreviewed SPIR-V execution mode')
            ids = {0}
        elif op == 71:
            require(len(a) >= 2 and a[1] in (11, 22, 41, 44, 4469, 4470, 5625, 5626),
                    'unreviewed SPIR-V decoration')
            if a[1] == 41:
                _, end = literal_string(a, 2); require(end + 1 == len(a), 'SPIR-V linkage extent')
            else:
                require(len(a) == (3 if a[1] in (11, 44, 5625) else 2), 'SPIR-V decoration extent')
            ids = {0}
        elif op in (19, 20, 21, 22): ids = {0}
        elif op == 23: ids = {0, 1}
        elif op == 32: ids = {0, 2}
        elif op in (33, 44, 57, 67, 70, 245):
            # OpInBoundsPtrAccessChain70 has the same ID operand schema as67;
            # retain the opcode itself,including its distinct inbounds promise.
            minimum = {33: 2, 44: 3, 57: 3, 67: 4, 70: 4, 245: 4}[op]
            require(len(a) >= minimum and (op != 245 or len(a) % 2 == 0), 'SPIR-V variable operand count')
            ids = set(range(len(a)))
        elif op == 43:
            require(len(a) in (3, 4), 'SPIR-V constant extent'); ids = {0, 1}
        elif op == 54: ids = {0, 1, 3}
        elif op == 59:
            require(len(a) in (3, 4), 'SPIR-V variable extent'); ids = {0, 1, *range(3, len(a))}
        elif op == 61:
            require(len(a) == 3 or (len(a) == 5 and a[3] == 2), 'unreviewed SPIR-V load memory operands')
            ids = {0, 1, 2}
        elif op in (79, 81, 82):
            end = 3 if op == 81 else 4
            require(len(a) > end, 'SPIR-V composite operands'); ids = set(range(end))
        elif op == 246:
            require(a[2] in (0, 1, 2), 'unreviewed SPIR-V loop control'); ids = {0, 1}
        elif op in (248, 249): ids = {0}
        elif op == 250:
            require(len(a) in (3, 5), 'SPIR-V branch extent'); ids = {0, 1, 2}
        else:
            require(op in typed and op in fixed, 'unreviewed SPIR-V opcode: ' + str(op))
            ids = set(range(len(a)))
        result = 1 if op in typed or op in (54, 59) else 0 if op in (11, 19, 20, 21, 22, 23, 32, 33, 248) else None
        require(all(0 < a[i] < w[3] for i in ids), 'SPIR-V ID bound')
        return ids, result
    records = []; names = {}; entries = {}; definitions = {}
    for op, a in operations:
        ids, result = schema(op, a); record = (op, a, ids, result); records.append(record)
        if result is not None:
            require(a[result] not in definitions, 'duplicate SPIR-V result ID'); definitions[a[result]] = record
        if op == 5:
            require(a[0] not in names, 'duplicate SPIR-V name target'); names[a[0]] = literal_string(a, 1)[0]
        if op == 15:
            require(a[1] not in entries, 'duplicate SPIR-V entry target'); entries[a[1]] = literal_string(a, 2)[0]
    labels = {}; functions = {}; globals_ = []; current = None
    for record in records:
        op, a, ids, result = record
        if op == 54:
            require(current is None, 'nested SPIR-V function')
            current = entries.get(a[1], names.get(a[1]))
            require(current is not None and current not in functions, 'unique named SPIR-V function required')
            functions[current] = []; labels[a[1]] = ('function', current)
        if current is None: globals_.append(record)
        else:
            if result is not None and op != 54: labels[a[result]] = ('local', current, len(functions[current]))
            functions[current].append(record)
            if op == 56: current = None
    require(current is None, 'unterminated SPIR-V function')
    visiting = set()
    def encode(record, defining=False):
        op, a, ids, result = record
        return (op, tuple(('result',) if defining and i == result else key(v) if i in ids else ('literal', v)
                          for i, v in enumerate(a)))
    def key(value):
        if value in labels: return labels[value]
        require(value in definitions and value not in visiting, 'undefined/cyclic SPIR-V global ID')
        visiting.add(value)
        definition = definitions[value]
        # Names distinguish addressable global objects;all their definitions,
        # types, initializers and annotations are still compared below.
        tag = ('named', names[value]) if definition[0] == 59 and value in names else (
            'global', hashlib.sha256(repr(encode(definition, True)).encode()).hexdigest())
        visiting.remove(value); labels[value] = tag; return tag
    for value in definitions: key(value)
    require(len(set(labels.values())) == len(labels), 'ambiguous SPIR-V global/local identity')
    normalized = (w[0:3], w[4], sorted((encode(v) for v in globals_), key=repr),
                  sorted((name, [encode(v) for v in body]) for name, body in functions.items()))
    return dict(sha256=hashlib.sha256(repr(normalized).encode()).hexdigest(),
        instructions=len(records), functions=len(functions), IDs=len(definitions),
        scope='All instructions/literals/decorations and in-function order preserved;only IDs/global/function order normalized.Not final ISA/runtime/performance.')


if __name__ == '__main__':
    raise SystemExit('Import embedded_dpas for linked-image checks;the closed comparison CLI is archived at f97893576704e6679c89ea6c4613d912665a0824.')
