"""Compile an offline line-table companion; do not load a GPU or replace objects."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import struct
import subprocess
import time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build = root/'build-sycl-upstream-jit'
out = base/'prefill-completion-source-map'
out.mkdir(mode=0o700)
source = root/'sycl/src/prefill/prefill.cpp'
original = build/'CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'
binary = base/'strata-prefill-expert-wait-candidate'
companion = out/'prefill-lines.o'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class Elf:
    def __init__(self, path):
        self.data = path.read_bytes()
        assert self.data[:6] == b'\x7fELF\x02\x01'
        header = struct.unpack_from('<HHIQQQIHHHHHH', self.data, 16)
        assert header[11] > 0 and header[12] > 0
        self.sections = [struct.unpack_from('<IIQQQQIIQQ', self.data,
                          header[5] + i*header[10]) for i in range(header[11])]
        strings = self.section_data(header[12])
        self.names = [self.string(strings, s[0]) for s in self.sections]

    @staticmethod
    def string(data, offset):
        return data[offset:data.index(b'\0', offset)].decode()

    def section_data(self, index):
        s = self.sections[index]
        return self.data[s[4]:s[4]+s[5]] if s[1] != 8 else b''

    def symbols(self, section=None):
        index = section if section is not None else self.names.index('.symtab')
        s = self.sections[index]
        strings = self.section_data(s[6])
        values = []
        for offset in range(s[4], s[4]+s[5], s[9]):
            symbol = struct.unpack_from('<IBBHQQ', self.data, offset)
            values.append({'name': self.string(strings, symbol[0]),
                           'info': symbol[1], 'other': symbol[2],
                           'section': symbol[3], 'value': symbol[4], 'size': symbol[5]})
        return values

    def executable(self):
        result = {}
        for i, s in enumerate(self.sections):
            if s[2] & 4:
                relocs = []
                for j, r in enumerate(self.sections):
                    if r[1] != 4 or r[7] != i:
                        continue
                    symbols = self.symbols(r[6])
                    for offset in range(r[4], r[4]+r[5], r[9]):
                        address, info, addend = struct.unpack_from('<QQq', self.data, offset)
                        sym = symbols[info >> 32]
                        section = (self.names[sym['section']] if sym['section'] < len(self.names)
                                   else str(sym['section']))
                        relocs.append([address, info & 0xffffffff, addend, sym['name'],
                                       section, sym['value'], sym['size'], sym['info'], sym['other']])
                result[self.names[i]] = {
                    'bytes': s[5], 'sha256': hashlib.sha256(self.section_data(i)).hexdigest(),
                    'relocations': relocs}
        return result

env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin')
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)
env.update(MKLROOT='/opt/intel/oneapi/mkl/2026.1',
           LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
record = {'scope': 'CPU-only line-table mapping of an actual frozen prefill wait; no GPU execution or production replacement',
          'passed': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256': digest(Path(__file__)), 'source_sha256': digest(source),
          'original_object_sha256': digest(original), 'binary_sha256': digest(binary),
          'observed_return_pc': '0x52ef0b', 'build_environment': {
              k: env.get(k) for k in ('PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'MKLROOT', 'LIBRARY_PATH')}}
start = time.monotonic()
try:
    assert record['source_sha256'] == 'c7ba0822a4f67b9a308eda8024d684e695c5b7ce5ba359143d790e2b9789243c'
    assert record['binary_sha256'] == 'a63f66eb289afa8b406434ccd7996b171fa93965e10b06eb7ba5decf528dab1c'
    build_record = json.loads((base/'prefill-expert-wait-build/record.json').read_text())
    assert build_record['passed']
    assert record['original_object_sha256'] == build_record['objects_after_build'][str(original.relative_to(build))]
    commands = subprocess.run(['/usr/bin/ninja', '-t', 'commands',
                               str(original.relative_to(build))], cwd=build, env=env,
                              check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()
    matches = [shlex.split(line) for line in commands if line.endswith(' -c '+str(source))]
    assert len(matches) == 1
    argv = matches[0]
    record['original_argv'] = argv.copy()
    for flag, value in [('-o', companion), ('-MF', out/'prefill-lines.d'), ('-MT', companion)]:
        assert argv.count(flag) == 1
        argv[argv.index(flag)+1] = str(value)
    argv.insert(1, '-gline-tables-only')
    record['companion_argv'] = argv
    with (out/'compile.stdout').open('wb') as stdout, (out/'compile.stderr').open('wb') as stderr:
        run = subprocess.run(argv, cwd=build, env=env, stdout=stdout, stderr=stderr, timeout=180)
    record['compile_exit_code'] = run.returncode
    assert run.returncode == 0
    assert digest(source) == record['source_sha256']
    assert digest(original) == record['original_object_sha256']
    record['production_object_unchanged'] = True
    record['companion_object_sha256'] = digest(companion)
    old = Elf(original)
    new = Elf(companion)
    exe = Elf(binary)
    old_exec, new_exec = old.executable(), new.executable()
    record['executable_sections_match'] = old_exec == new_exec
    record['original_executable_sections'] = old_exec
    record['companion_executable_sections'] = new_exec
    assert old_exec == new_exec, 'CPU machine code or canonical relocation targets differ; source mapping rejected'
    candidates = [s for s in exe.symbols() if 'run_impl' in s['name'] and (s['info'] & 15) == 2
                  and s['value'] <= int(record['observed_return_pc'], 16) < s['value']+s['size']]
    assert len(candidates) == 1
    actual = candidates[0]
    demangled = subprocess.run(['/usr/bin/c++filt', actual['name']], capture_output=True,
                               text=True, check=True, timeout=5).stdout.strip()
    assert '{lambda()#15}::operator()() const' in demangled
    old_symbol = [s for s in old.symbols() if s['name'] == actual['name']]
    new_symbol = [s for s in new.symbols() if s['name'] == actual['name']]
    assert len(old_symbol) == len(new_symbol) == 1
    old_symbol, new_symbol = old_symbol[0], new_symbol[0]
    assert old_symbol == new_symbol
    assert actual['size'] == old_symbol['size']
    offset = int(record['observed_return_pc'], 16)-actual['value']
    record.update(symbol=actual['name'], demangled_symbol=demangled,
                  binary_symbol_base=hex(actual['value']), function_bytes=actual['size'],
                  offset_in_function=offset, object_symbol_value=hex(new_symbol['value']))
    section = new.names[new_symbol['section']]
    # A stack return PC names the instruction after the call; map it and PC-1.
    addresses = [hex(new_symbol['value']+offset), hex(new_symbol['value']+offset-1)]
    line_argv = ['/usr/bin/addr2line', '-C', '-f', '-i', '-e', str(companion), '-j', section]+addresses
    lines = subprocess.run(line_argv, capture_output=True, text=True, check=True, timeout=10)
    record.update(addr2line_argv=line_argv, addr2line_output=lines.stdout,
                  addr2line_stderr=lines.stderr, passed=True)
    (out/'source-lines.txt').write_text(lines.stdout)
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record['elapsed_seconds'] = time.monotonic()-start
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k: v for k, v in record.items() if not k.endswith('executable_sections')}, indent=2))
