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
def embedded_dpas(blob):
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
            if not any(text==KERNEL_ENTRY for _,_,text in entries):continue
            require(len(entries)==1 and entries[0][0]==6 and entries[0][1] in functions,
                    'one exact C4 kernel entrypoint')
            reachable=set();pending=[entries[0][1]];edges={}
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
    require(len(matches)==1, 'one exact linked C4 device image')
    return dict(matches[0],device_images=len(seen),
        scope='Reachable static GenX DPAS calls in exact linked ELF SPIR-V;not final native ISA,utilization or dynamic call count.')


if __name__ == '__main__':
    raise SystemExit('Import embedded_dpas for linked-image checks;the closed comparison CLI is archived at f97893576704e6679c89ea6c4613d912665a0824.')
