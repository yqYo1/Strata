"""Shared bounded C++ steady-clock qualification, no GPU/runtime calls.

Repository LICENSE. Reuses the measured phase-prefill clock contract; provider
identity, headers/tools and actual native imports are checked prospectively.
"""
from pathlib import Path
import re
import shlex
import subprocess
import build_engine as build
import measurement_guards as guards
from mtp_resident_control import fingerprint

P = build.P
SOURCE = P / 'tools/sycl/clock_overhead.cpp'
MIN_TIME_MS = 10000
MAX_FRACTION = .0001


def prepare(native, compiler, environment, root, provider):
    env = {k: environment[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','PYTHONDONTWRITEBYTECODE')}
    command = [compiler['path'], '-std=c++17', '-O2', str(SOURCE), '-o', str(root/'clock_probe')]
    raw = subprocess.check_output([compiler['path'],'-std=c++17','-M',str(SOURCE)],env=env,text=True,timeout=30)
    inputs = {str(Path(p).resolve()):build.digest(p) for p in shlex.split(raw.replace('\\\n',' '))[1:]}
    inputs[compiler['path']] = compiler['sha256']; inputs[provider['path']] = provider['sha256']
    tools = {n:str(Path('/usr/bin',n).resolve()) for n in ('nm','objdump')}
    for name in ('ld','as'):
        path = subprocess.check_output([compiler['path'],'-print-prog-name='+name],env=env,text=True,timeout=15).strip()
        tools[name] = str((Path(path) if '/' in path else Path('/usr/bin',path)).resolve())
    for path in tools.values(): inputs[path] = build.digest(path)
    symbol = '_ZNSt6chrono3_V212steady_clock3nowEv'
    guards.require(symbol in subprocess.check_output([tools['nm'],'-D',native['path']],env=env,text=True,timeout=15),
        'native steady-clock import missing')
    assembly = subprocess.check_output([tools['objdump'],'--disassemble='+symbol,provider['path']],env=env,text=True,timeout=15)
    guards.require(re.search(r'mov\s+\$0x1,%edi',assembly) and '<clock_gettime@plt>' in assembly,
        'clock provider is not the inspected CLOCK_MONOTONIC implementation')
    row = dict(inputs=inputs,inputs_sha256=fingerprint(inputs),command=command,environment=env,
        environment_sha256=fingerprint(env),
        provider=provider,compiler=compiler,minimum_time_ms=MIN_TIME_MS,maximum_endpoint_fraction=MAX_FRACTION,
        source_sha256=build.digest(SOURCE),source_contract='Same native libstdc++ steady_clock::now;65536 consecutive pairs,sort/output outside pairs. Observed endpoint/rounding screen,not a universal bound or time subtraction.')
    check(row); return row


def check(row):
    guards.require(fingerprint(row['inputs']) == row['inputs_sha256'] and
        fingerprint(row['environment']) == row['environment_sha256'] and
        row['minimum_time_ms'] == MIN_TIME_MS and row['maximum_endpoint_fraction'] == MAX_FRACTION,
        'clock input seal/criteria changed')
    for path, sha in row['inputs'].items(): guards.require(build.digest(path) == sha, 'clock input changed: '+path)


def compact(row):
    return {k:v for k,v in row.items() if k not in ('inputs','environment')}


def evidence(text, provider):
    guards.require(len(text.splitlines()) == 2,'unexpected clock output')
    names = re.findall(r'^clock_provider=(.+)$',text,re.M)
    match = re.search(r'^clock_samples=(\d+) steady=(\d+) period_num=(\d+) period_den=(\d+) '
        r'resolution_ns=(\d+) min_ns=(\d+) p99_ns=(\d+) max_ns=(\d+)$',text,re.M)
    guards.require(len(names) == 1 and match and Path(names[0]).resolve() == Path(provider['path']).resolve() and
        build.digest(names[0]) == provider['sha256'],'actual clock provider/record differs')
    samples,steady,num,den,res,minimum,p99,maximum = map(int,match.groups())
    guards.require((samples,steady,num,den) == (65536,1,1,1000000000) and 0 < res <= 1000 and
        0 <= minimum <= p99 <= maximum,'clock contract/conservation differs')
    fraction = (2*maximum/1e6+.05)/MIN_TIME_MS
    guards.require(fraction <= MAX_FRACTION,'clock endpoint/rounding budget exceeded')
    return dict(samples=samples,resolution_ns=res,min_ns=minimum,p99_ns=p99,max_ns=maximum,
        endpoint_rounding_fraction_at_floor=fraction,provider=provider)
