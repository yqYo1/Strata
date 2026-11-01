"""Successful UR kernel creation/name/launch provenance. Repository LICENSE (MIT).

Extracted from the closed Main-authored GDN storage fixture; no GPU/runtime
imports. Native imports retain their own identity even before a name query.
"""
import re


def launch_census(text):
    handles, launches, unmapped, serial = {}, {}, 0, 0
    for line in text.splitlines():
        if 'UR_RESULT_SUCCESS' not in line: continue
        if '<--- urKernelCreate(' in line:
            v = re.search(r'\.pKernelName = .*?\(([^()]+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
            if v: handles[v[2]] = (v[1], 'named', v[1])
        elif '<--- urKernelCreateWithNativeHandle(' in line:
            # oneMKL imports precompiled native kernels without pKernelName.
            # Successful import is provenance; do not invent an absent name.
            v = re.search(r'\.hNativeKernel = (0x[0-9a-f]+).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
            if v:
                serial += 1; handles[v[2]] = ('native_import_' + str(serial), 'native_import', None)
        elif '<--- urKernelGetInfo(' in line and 'UR_KERNEL_INFO_FUNCTION_NAME' in line:
            handle = re.search(r'\.hKernel = (0x[0-9a-f]+)', line)
            value = re.search(r'\.pPropValue = 0x[0-9a-f]+ \(([^()]+)\)', line)
            if handle and value and handle[1] in handles:
                label, origin, _ = handles[handle[1]]
                handles[handle[1]] = (label, origin, value[1])
                if label in launches: launches[label]['name'] = value[1]
        if not re.search(r'<--- urEnqueueKernelLaunch(?:WithArgsExp|CustomExp)?\(', line): continue
        v = re.search(r'\.hKernel = (0x[0-9a-f]+)', line)
        if not v or v[1] not in handles: unmapped += 1; continue
        label, origin, name = handles[v[1]]
        group = launches.setdefault(label, dict(creation=origin, name=name, launches=0))
        group['launches'] += 1
    if not launches or unmapped:
        raise RuntimeError('complete successful GPU kernel census required')
    return dict(launches=sum(v['launches'] for v in launches.values()), per_kernel=launches, unmapped=unmapped)
