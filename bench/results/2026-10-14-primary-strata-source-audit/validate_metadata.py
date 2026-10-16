#!/usr/bin/env python3
"""Validate only the actual primary's project metadata under the shared lock."""
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
Z = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/feat-layer-major-bulk-upload-20261013')
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
L = T.parents[2] / '.lint'
O = P / 'bench/results/2026-10-14-primary-strata-source-audit'
env = json.loads((B / 'main-layer-major-bulk-full-lints-root-v3/record.json').read_text())['environment']
env = {k: v for k, v in env.items() if k not in ['STRATA_LINT_BUILD']}

def main():
    record = {'passed': False, 'gpu_executed': False, 'engine_source_modified': False,
              'scope': 'primary project skill, links, audit metadata; no Xe universal build/release policy', 'commands': []}
    def run(argv):
        p = subprocess.run([str(x) for x in argv], cwd=P, env=env, text=True, capture_output=True, timeout=60)
        record['commands'].append({'argv': [str(x) for x in argv], 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
        assert p.returncode == 0, (argv, p.stdout[-3000:], p.stderr[-3000:])
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run([L / 'venv/bin/python', '/home/yayoi/.codex/skills/.system/skill-creator/scripts/quick_validate.py', P / '.agents/skills/strata-upstream-sync'])
            assert (P / '.agents/skills').is_symlink()
            assert (P / '.agents/skills').readlink() == Path('../.claude/skills')
            assert (P / '.agents/skills/strata-upstream-sync').resolve() == P / '.claude/skills/strata-upstream-sync'
            assert not Path('/home/yayoi/.agents/skills/strata-upstream-sync').exists()
            assert not Path('/home/yayoi/.codex/skills/strata-upstream-sync').exists()
            run(['git', 'diff', '--check'])
            run(['git', 'diff', '--exit-code', '--', 'src', 'include', 'cmake', 'CMakeLists.txt', 'serve', 'setup.py'])
            files = [P / 'AGENTS.md', *sorted((P / '.claude/skills/strata-upstream-sync').rglob('*')), *sorted(O.rglob('*'))]
            files = [x for x in files if x.is_file() and x.name != 'metadata-validation.json']
            md = [x for x in files if x.suffix == '.md']
            run(['mdl', '--style', Z / 'tools/lint/mdl_style.rb', *md])
            run([L / 'bin/lychee', '--offline', '--no-progress', *md])
            run([L / 'venv/bin/codespell', '--config', Z / 'tools/lint/codespell.cfg', *md])
            for path in files:
                run([L / 'bin/gitleaks', 'detect', '--no-git', '--redact', '--no-banner', '--exit-code', '1', '--source', path])
            run([L / 'venv/bin/python', '-c', "import yaml; from pathlib import Path; d=yaml.safe_load(Path('.claude/skills/strata-upstream-sync/agents/openai.yaml').read_text()); assert d['interface']['short_description']; print('UI YAML valid')"])
            record.update({'passed': True, 'relative_discovery': 'passed', 'no_global_install': 'passed', 'primary_source_unchanged': 'passed',
                           'independent_forward_reports': ['R445', 'R446'], 'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        except Exception as e:
            record['error'] = repr(e)
            raise
        finally:
            (O / 'metadata-validation.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'passed': record['passed'], 'checks': len(record['commands'])}))

if __name__ == '__main__':
    main()
