"""Own MIT6e40 QSA: source-only independent K/V/probability SLM pitches.

Keep32cells,global loads,arithmetic,barriers and WG128/SG16 unchanged. Padded
leading dimensions require separate configured-SDK admission before compilation
or GPU execution. No production/decode route is changed by this generator.
"""
import hashlib

BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
PREFIX_SHA256 = '4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique pitch overlay: '+before[:100])
    return text.replace(before, after, 1)


def payload(k, v, p):
    # Eight active local arguments;the unused Q accessor adds4source bytes.
    return max(k*256,32*v,16*256)*2 + 16*p*2 + (16*33+3*16)*4 + 32*8 + 16*4


def generated(project, pitches=(40,264,40)):
    k,v,p = pitches
    if not all(type(n) is int and low <= n <= high and n % 8 == 0
               for n,low,high in ((k,32,256),(v,256,1024),(p,32,256))):
        raise ValueError('bounded whole-element pitches,at least logical extent,multiple8 required')
    raw = (project/PREFIX).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFIX_SHA256:
        raise ValueError('generate pitch proposal only from immutable6e40prefix')
    source = raw.decode();body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    wrapper = source[source.index('bool '+EXPORT+'('):source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    header = ('// Main-authored component declarations;repository LICENSE.\n'
              '#pragma once\n#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <sycl/sycl.hpp>\n')
    outputs = {}
    for name,values in (('pitch_control',(32,256,32)),('pitch_candidate',pitches)):
        kp,vp,pp = values;extra = payload(kp,vp,pp)-payload(32,256,32)
        namespace = 'namespace strata::kernels::'+name+' {'
        edits = [
            ('namespace strata::kernels {',namespace),
            ('Workgroup = 128;', 'Workgroup = 128;\n'
             f'constexpr int KPitch = {kp}, VPitch = {vp}, PPitch = {pp};\n'
             'constexpr int KvWords = std::max({KPitch * Dim, Cells * VPitch, 16 * Dim});\n'
             'static_assert(KPitch >= Cells && VPitch >= Dim && PPitch >= Cells);\n'
             'static_assert(KPitch % 8 == 0 && VPitch % 8 == 0 && PPitch % 8 == 0);'),
            ('storage_contract<Cells * Dim>()','storage_contract<KvWords>()'),
            ('storage_contract<16 * Cells>()','storage_contract<16 * PPitch>()'),
            ('(16 * Dim + Cells * Dim + 16 * Cells) * 2','(16 * Dim + KvWords + 16 * PPitch) * 2'),
            ('LocalPayload<Cells> == 28160',f'LocalPayload<Cells> == {28160+extra}'),
            ('LocalPayload<Cells + 1> == 28224',f'LocalPayload<Cells + 1> == {28224+extra}'),
            ('LocalPayload<Cells + 1, true> == 20036',f'LocalPayload<Cells + 1, true> == {20036+extra}'),
            ('QsaPrefixStateProbe probe = {}) {','QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr) {'),
            ('static_assert(Dim % 4 == 0);','static_assert(Dim % 4 == 0);\n'
             '  static_assert(OrderedExp && CacheQueries && VectorK4 && VectorV4 && !TilePacked && !CaptureState);'),
            ('MatrixTile<Cells * Dim>, 1> kvstorage','MatrixTile<KvWords>, 1> kvstorage'),
            ('MatrixTile<16 * Cells>, 1> pstorage','MatrixTile<16 * PPitch>, 1> pstorage'),
            ('kv[(d + j) * Cells + cell]','kv[(d + j) * KPitch + cell]'),
            ('kv + d * Cells + cell, Cells','kv + d * KPitch + cell, KPitch'),
            ('probability[h * Cells + c0]','probability[h * PPitch + c0]'),
            ('probability[h * Cells + c1]','probability[h * PPitch + c1]'),
            ('else probability[tid * Cells + c]','else probability[tid * PPitch + c]'),
            ('kv[cell * Dim + d + j]','kv[cell * VPitch + d + j]'),
            ('probability + head * Cells + c, Cells','probability + head * PPitch + c, PPitch'),
            ('kv + c * Dim + dim_base + j * 16, Dim','kv + c * VPitch + dim_base + j * 16, VPitch'),
            ('  sycl_backend::finish(stream, event);\n  return true;',
             '  if (submitted) *submitted = event;\n  sycl_backend::finish(stream, event);\n  return true;')]
        value = body
        for before,after in edits:
            value = once(value,before,after)
        restored = value
        for before,after in reversed(edits):
            restored = once(restored,after,before)
        if restored != body:
            raise ValueError('pitch overlay changed an undeclared source byte')
        arm = wrapper
        wrapper_edits = [('void *stream) {','void *stream, sycl::event *submitted) {'),
            ('output, queries, bounds, live_cells, scratch, stream);',
             'output, queries, bounds, live_cells, scratch, stream, {}, submitted);')]
        for before,after in wrapper_edits:
            arm = once(arm,before,after)
        restored = arm
        for before,after in reversed(wrapper_edits):
            restored = once(restored,after,before)
        if restored != wrapper:
            raise ValueError('pitch wrapper changed an undeclared source byte')
        outputs[name+'.cpp'] = (value+arm+'}\n').encode()
        header += namespace+'\n'+arm[:arm.index(' {')]+';\n}\n'
    outputs['qsa-slm-pitch.hpp'] = header.encode()
    return outputs


def host_controls(project):
    """Enumerate distinct writer/reader schedules;no compiler or device calls."""
    reference = None;cases = 0
    for kp in (32,40,48):
        for vp in (256,264,272):
            for pp in (32,40,48):
                outputs = generated(project,(kp,vp,pp));control = outputs['pitch_control.cpp']
                if reference is None:
                    reference = control
                if reference != control:
                    raise ValueError('pitch choices changed the control')
                # Writers follow128work-items with four-value stores. Readers
                # independently enumerate the8x16/16x16matrix tiles.
                kw = {};vw = {};pw = {}
                for tid in range(128):
                    for i in range(tid,32*64,128):
                        cell,d = i%32,(i//32)*4
                        for j in range(4):kw[d+j,cell] = (d+j)*kp+cell
                        cell,d = i//64,(i%64)*4
                        for j in range(4):vw[cell,d+j] = cell*vp+d+j
                for h in range(16):
                    for lane in range(16):
                        for c in (lane,lane+16):pw[h,c] = h*pp+c
                kr = {(d+r,c+s):(d+r)*kp+c+s for d in range(0,256,16)
                      for c in (0,16) for r in range(16) for s in range(16)}
                vr = {(c+r,d+j*16+s):(c+r)*vp+d+j*16+s for c in (0,16)
                      for r in range(16) for d in (0,64,128,192) for j in range(4) for s in range(16)}
                pr = {(h+r,c+s):(h+r)*pp+c+s for h in (0,8)
                      for r in range(8) for c in (0,16) for s in range(16)}
                words = max(kp*256,32*vp,16*256)
                for writer,reader,size,count in ((kw,kr,words,8192),(vw,vr,words,8192),(pw,pr,16*pp,512)):
                    if writer != reader or len(writer) != count or len(set(writer.values())) != count or not all(0 <= v < size for v in writer.values()):
                        raise ValueError('uninitialized,aliased or out-of-allocation matrix operand')
                cases += 1
    rejected = 0
    for pitches in ((31,256,32),(32,255,32),(32,256,31),(33,256,32),(40,264,33),(264,264,40)):
        try:
            generated(project,pitches)
        except ValueError:
            rejected += 1
        else:
            raise ValueError('invalid pitch accepted')
    if payload(32,256,32) != 20032 or payload(40,264,40) != 24384:
        raise ValueError('local payload accounting differs')
    return dict(passed=True,layout_cases=cases,rejected_invalid_pitches=rejected,
        active_local_bytes=[20032,24384],GPU_executed=False,compiler_executed=False,
        SDK_legality_qualified=False,numerical_qualified=False)


if __name__ == '__main__':
    import json
    from pathlib import Path
    print(json.dumps(host_controls(Path(__file__).resolve().parents[2])))
