"""Bounded CPU-only parser for the frozen static 48x512 route-pair diagnostic.

FNV identifies the host slot map for joining; it is neither a math digest nor
proof of weight/model identity. Caller separately pins source/profile/model.
"""
import re
import struct

PREFIX = b'CACHE_ROUTE_PAIRS_V1 '
REQUEST_LIMIT = 4 * 1024**2
JOB_LIMIT = 16 * 1024**2
TEXT_LIMIT = 64 * 1024**2
U64 = 2**64 - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(text):
    require(re.fullmatch(r'0|[1-9][0-9]{0,19}', text) is not None, 'noncanonical uint64')
    value = int(text)
    require(value <= U64, 'uint64 overflow')
    return value


def fields(line, kind, names):
    words = line.split(' ')
    require(words[:2] == ['CACHE_ROUTE_PAIRS_V1', kind] and len(words) == len(names)+2,
            'wrong frame type/field count')
    result = {}
    for word, name in zip(words[2:], names):
        key, sep, value = word.partition('=')
        require(sep and key == name, 'unknown/duplicate/out-of-order field')
        if name.endswith('fnv1a64'):
            require(re.fullmatch('[0-9a-f]{16}', value) is not None, 'noncanonical FNV')
            result[name] = value
        else:
            result[name] = integer(value)
    return result


def static_profile(data):
    """Independently derive current admission's fixed slot map from STRP v1.

    Only a complete pinned profile is accepted, including its inverse-table tail.
    """
    require(len(data) == 196632 and data[:4] == b'STRP', 'profile size/magic')
    require(struct.unpack_from('<5I', data, 4) == (1,48,512,24576,24576), 'profile geometry')
    ranked = list(struct.iter_unpack('<HH', data[24:24+24576*4]))
    require(len(set(ranked)) == 24576 and all(l<48 and e<512 for l,e in ranked), 'profile bounds/duplicates')
    inverse = struct.unpack_from('<24576i', data, 24+24576*4)
    # Tail is indexed by layer/expert and stores the rank of that pair.
    require(all(inverse[l*512+e] == rank for rank,(l,e) in enumerate(ranked)), 'profile inverse rank table')
    slots = [-1]*24576
    next_slot = [l*2 for l in range(48)]
    for layer, expert in ranked:
        end = 128 if layer == 47 else (layer+1)*2
        if next_slot[layer] < end:
            slots[layer*512+expert] = next_slot[layer]
            next_slot[layer] += 1
    residents = {(i//512,i%512) for i,value in enumerate(slots) if value>=0}
    require(len(residents)==128, 'current cache must have exactly 128 resident pairs')
    fingerprint = 14695981039346656037
    for value in slots:
        for byte in struct.pack('<i', value):
            fingerprint = ((fingerprint ^ byte)*1099511628211) & U64
    return residents, f'{fingerprint:016x}'


def parse_request(data, ordinal, done, residents, expected_fingerprint):
    require(isinstance(data, bytes) and len(data)<=TEXT_LIMIT, 'interval text budget/type')
    require(data.endswith(b'\n'), 'incomplete request stderr interval')
    require(1<=ordinal<=4, 'ordinal outside four-request job')
    words = done.split()
    require(len(words)==16 and words[0]=='DONE' and words[5] in ('length','stop'), 'incomplete/cancelled DONE')
    require(integer(words[1])==64 and integer(words[2])==32768 and integer(words[8])==0 and
            integer(words[14])==32768, 'not a complete fresh64/32K request')
    done_hits, done_look, done_offload = (integer(words[i]) for i in (9,10,15))
    lines=[];frame_bytes=0
    for raw in data.splitlines(keepends=True):
        require(len(raw)<=65536 and raw.endswith(b'\n'), 'oversized/incomplete stderr line')
        if b'CACHE_ROUTE_PAIRS_V1' in raw:
            require(raw.startswith(PREFIX), 'misplaced/noncanonical frame prefix')
            require(not raw.endswith(b'\r\n'), 'noncanonical frame newline')
            try: line=raw[:-1].decode('ascii')
            except UnicodeDecodeError as error: raise ValueError('non-ASCII frame') from error
            frame_bytes += len(raw)
            require(frame_bytes<=REQUEST_LIMIT, 'request pair output exceeds 4 MiB')
            lines.append(line)
    require(4<=len(lines)<=24579, 'missing/empty/oversized frame')
    begin=fields(lines[0],'BEGIN',('request','layers','experts','enabled','supported','truncated','error','complete'))
    residency=fields(lines[1],'RESIDENCY',('request','cells','unchanged','start_fnv1a64','end_fnv1a64'))
    end=fields(lines[-1],'END',('request','cells','callbacks','entries','hits','refused','offloaded','callback_pairs',
                            'done_hits','done_look','done_offload','reconciled','complete'))
    require(begin==dict(request=ordinal,layers=48,experts=512,enabled=1,supported=1,truncated=0,error=0,complete=1),
            'unsupported/error/incomplete/incorrect BEGIN')
    require(residency==dict(request=ordinal,cells=24576,unchanged=1,start_fnv1a64=expected_fingerprint,
                           end_fnv1a64=expected_fingerprint), 'resident identity changed/not pinned static map')
    require(end['request']==ordinal and end['complete']==end['reconciled']==1, 'END incomplete/mismatched ordinal')
    require(0<end['callbacks']<=U64 and end['callbacks']%48==0, 'invalid complete-layer callback count')
    totals=dict(entries=0,hits=0,refused=0,offloaded=0,callback_pairs=0)
    pairs=[];previous=(-1,-1);layer_entries=[0]*48
    for line in lines[2:-1]:
        w=line.split(' ')
        require(w[:2]==['CACHE_ROUTE_PAIRS_V1','PAIR'] and len(w)==9, 'unexpected/duplicate frame marker or PAIR')
        layer,expert,entries,hits,refused,offloaded,callback_pairs = map(integer,w[2:])
        key=(layer,expert)
        require(layer<48 and expert<512 and key>previous, 'unsorted/duplicate/out-of-range pair')
        previous=key
        require(entries>0 and entries==hits+refused+offloaded and 0<callback_pairs<=entries and
                callback_pairs<=end['callbacks']//48, 'invalid pair classification/callback units')
        # The supported spec4 split callbacks have NT at most 2. Repeated IDs can
        # occur within one token's k=10, so do not assume entries<=2*callback_pairs.
        require(offloaded==0, 'fixed no-offload profile changed')
        require((hits==entries and refused==0) if key in residents else (hits==0 and refused==entries),
                'classification inconsistent with fixed pinned resident set')
        values=dict(entries=entries,hits=hits,refused=refused,offloaded=offloaded,callback_pairs=callback_pairs)
        for name,value in values.items():
            totals[name]+=value
            require(totals[name]<=U64, 'sparse total overflow')
        layer_entries[layer]+=entries
        pairs.append(dict(layer=layer,expert=expert,**values))
    require(len(pairs)==end['cells'] and 0<len(pairs)<=24576, 'sparse count mismatch')
    require(all(totals[name]==end[name] for name in totals), 'sparse sums differ from END')
    require(all(value==layer_entries[0] and value>0 and value%10==0 for value in layer_entries),
            'missing/unequal layer route coverage or wrong top-k10')
    require(totals['entries']==totals['hits']+totals['refused']==done_look and
            totals['hits']==done_hits and totals['offloaded']==done_offload==0 and
            (end['done_hits'],end['done_look'],end['done_offload'])==(done_hits,done_look,done_offload),
            'DONE reconciliation/admission/offload mismatch')
    return dict(passed=True,ordinal=ordinal,begin=begin,residency=residency,end=end,pairs=pairs,
                frame_bytes=frame_bytes,layer_entries=layer_entries,
                semantics='main target evaluated route entries including speculative rejects; callback_pairs are distinct experts per callback, not CPU/native jobs or traffic')
