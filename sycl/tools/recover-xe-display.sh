#!/bin/bash
# Internal GUI handoff. The human entry point remains strata-gpu-recover.
set -eu
exec /usr/bin/python3 - "$(readlink -f -- "$0")" "$@" <<'PY'
import contextlib
import datetime
import json
import os
from pathlib import Path
import pwd
import re
import sys
import tempfile
import time
import types

BASE = Path('/var/log/strata-gpu-recovery')
SERVICE = 'sddm.service'
BDF = '0000:05:00.0'


def load_core(directory):
    path = directory / 'strata-xe-recover-core'
    source = path.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    module = types.ModuleType('strata_xe_recovery')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def secure_base():
    os.umask(0o077)
    BASE.mkdir(parents=True, exist_ok=True)
    info = BASE.stat()
    if BASE.is_symlink() or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise RuntimeError('Unsafe recovery log directory')


def save_plan(path, plan):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(path)


def read_plan(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink() or path.parent.parent != BASE:
        raise RuntimeError('Invalid recovery plan location')
    if path.parent.stat().st_uid != os.geteuid() or path.parent.stat().st_mode & 0o077:
        raise RuntimeError('Unsafe recovery plan directory')
    if path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077:
        raise RuntimeError('Unsafe recovery plan ownership/permissions')
    plan = json.loads(path.read_text())
    if plan.get('approved') is not True or plan.get('service') != SERVICE or plan.get('bdf') != BDF:
        raise RuntimeError('Display shutdown was not approved')
    if plan.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Recovery plan belongs to another boot')
    account = pwd.getpwnam(plan['user'])
    if account.pw_uid == 0 or account.pw_uid != plan['uid']:
        raise RuntimeError('Invalid ordinary health-probe account')
    return plan


def reject_pending(core):
    path = BASE / 'stalled-writer.json'
    if not path.exists():
        return
    old = json.loads(path.read_text())
    try:
        raw = Path(f'/proc/{old["pid"]}/stat').read_text()
        ticks = int(raw[raw.rfind(')') + 2:].split()[19])
    except FileNotFoundError:
        return
    if ticks == old['start_ticks']:
        raise core.RecoveryRefused('A previous kernel writer remains; display shutdown refused')


def independent_display():
    for card in Path('/sys/class/drm').glob('card[0-9]*'):
        if not re.fullmatch(r'card\d+', card.name) or (card / 'device').resolve().name == BDF:
            continue
        boot = card / 'device/boot_vga'
        if boot.exists() and boot.read_text().strip() == '1':
            if any(p.read_text().strip() == 'connected' for p in card.parent.glob(card.name + '-*/status')):
                return True
    return False


def owned_by_display_service(client):
    groups = Path(f'/proc/{client["pid"]}/cgroup').read_text().splitlines()
    return '0::/system.slice/' + SERVICE in groups


def admissible(core, runner, expected=None):
    reject_pending(core)
    device = core.Device(BDF)
    description = device.describe()
    if description['driver'] != 'xe' or description['boot_vga'] != '0' or any(x == 'connected' for x in description['connectors'].values()):
        raise core.RecoveryRefused('Target GPU is bound incorrectly or drives a display')
    if description['survivability_mode'] not in ('unavailable', '0') or not independent_display():
        raise core.RecoveryRefused('No independent boot/display GPU, or firmware survivability is active')
    clients, hidden = device.clients()
    if hidden or not clients or any(c['comm'] != 'Xorg' or c['state'] == 'D' for c in clients):
        raise core.RecoveryRefused('Owners are not exclusively inspectable Xorg clients')
    for client in clients:
        if not owned_by_display_service(client):
            raise core.RecoveryRefused('Xorg is not owned by the known display service')
    identities = sorted((c['pid'], c['start_ticks']) for c in clients)
    if expected is not None and identities != [tuple(x) for x in expected]:
        raise core.RecoveryRefused('GPU owners changed since approval; no display shutdown')
    state = runner.run('display-before', ['/usr/bin/systemctl', 'is-active', SERVICE], seconds=5).strip()
    if state != 'active':
        raise core.RecoveryRefused('Display service is no longer active')
    return identities


def confirm():
    try:
        # BufferedRandom (r+) requires seeking, which a real terminal cannot do.
        with open('/dev/tty', 'r', encoding='utf-8') as reader, open('/dev/tty', 'w', encoding='utf-8') as writer:
            writer.write('XorgがB570を使用中です。GUIを終了してGPU復旧を試します。\n'
                           '開いているGUIアプリも終了します。未保存の作業を先に保存してください。\n'
                           '処理後はログイン画面の起動を試みます。復旧できない場合は戻らないことがあります。\n'
                           'GUIを終了して続ける場合だけ yes と入力してください（Enterで中止）: ')
            writer.flush()
            return reader.readline(32).strip() == 'yes'
    except OSError:
        return False


def unit_arguments(script, path, user):
    # ExecStopPost uses systemd's command parser, not a shell. Restrict the
    # installed paths so no quoting/variable expansion ambiguity is introduced.
    if not all(re.fullmatch(r'[A-Za-z0-9_./-]+', str(p)) for p in [script, path]):
        raise RuntimeError('Unsupported recovery helper path')
    unit = 'strata-gpu-recover-' + str(os.getpid()) + '-' + str(time.monotonic_ns())
    return ['/usr/bin/systemd-run', '--unit=' + unit, '--collect', '--service-type=exec',
            '--expand-environment=no', '--property=RuntimeMaxSec=12min',
            '--property=TimeoutStopSec=45s', '--property=Restart=no',
            '--property=ExecStopPost=/bin/bash ' + str(script) + ' --restore ' + str(path),
            '--setenv=SUDO_USER=' + user, '/bin/bash', str(script), '--worker', str(path)]


def offer(core, script):
    secure_base()
    user = os.environ.get('SUDO_USER', '')
    account = pwd.getpwnam(user)
    if account.pw_uid == 0:
        raise core.RecoveryRefused('Use the recovery command from your ordinary account')
    health = script.parent / 'strata-xe-health'
    if not health.is_file() or not os.access(health, os.X_OK):
        raise core.RecoveryRefused('Health probe is unavailable')
    output = Path(tempfile.mkdtemp(prefix='display-', dir=BASE))
    runner = core.Runner(output)
    with core.acquire_recovery_lock(BASE):
        identities = admissible(core, runner)
        if not confirm():
            print('GUIを終了せず中止しました。リセットは実行していません。')
            return 4
        plan = dict(approved=True, service=SERVICE, bdf=BDF, user=user, uid=account.pw_uid,
                    clients=identities, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    approved_monotonic=time.monotonic(), worker_started=False,
                    display_stop_attempted=False, display_restore_requested=False)
        path = output / 'plan.json'
        save_plan(path, plan)
    print('GUIの外で復旧処理を開始します。結果は通常の latest.json と以下のログに保存します。', flush=True)
    print('ログ: ' + str(output), flush=True)
    pending = dict(action='display-recovery', status='queued', healthy=False,
                   recovery_completed=False, dump_saved=False, steps=runner.calls)
    (output / 'report.json').write_text(json.dumps(pending, indent=2) + '\n')
    core.publish_summary(pending, output, runner)
    runner.run('launch-service', unit_arguments(script, path, user), seconds=10)
    return 3  # Handoff accepted; this is not a GPU health result.


def restore(core, path, plan=None, runner=None, held_lock=None):
    plan = read_plan(path) if plan is None else plan
    if not plan['display_stop_attempted'] or plan['display_restore_requested']:
        return
    with core.acquire_recovery_lock(BASE, held_lock):
        # The backup runs after the worker exits. Serialize its display startup
        # with every reset, and do not reopen xe while a writer remains stuck.
        plan = read_plan(path)
        if plan['display_restore_requested']:
            return
        reject_pending(core)
        runner = core.Runner(path.parent) if runner is None else runner
        runner.run('display-start', ['/usr/bin/systemctl', 'start', SERVICE], seconds=30)
        plan['display_restore_requested'] = True
        save_plan(path, plan)


def restore_after_exit(core, path):
    plan, runner = read_plan(path), core.Runner(path.parent)
    error = None
    try:
        restore(core, path, plan, runner)
    except Exception as failure:
        error = str(failure)
    report_path = path.parent / 'report.json'
    receipt = json.loads(report_path.read_text()) if report_path.exists() else {}
    if not receipt.get('finished_utc'):
        receipt.update(action='display-recovery', status='interrupted', healthy=False,
                       recovery_completed=False, dump_saved=False,
                       error=error or 'Worker exited before publishing a final health result',
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        report_path.write_text(json.dumps(receipt, indent=2) + '\n')
        os.environ['SUDO_USER'] = plan['user']
        core.publish_summary(receipt, path.parent, runner)
    if error:
        raise core.RecoveryError(error)


def worker(core, script, path):
    plan = read_plan(path)
    runner = core.Runner(path.parent)
    receipt = dict(action='display-recovery', status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   healthy=False, recovery_completed=False, dump_saved=False, attempts=[], steps=runner.calls)
    code = 1
    with core.acquire_recovery_lock(BASE) as lock:
        try:
            if plan['worker_started'] or time.monotonic() - plan['approved_monotonic'] > 60:
                raise core.RecoveryRefused('Recovery approval is stale or already used')
            admissible(core, runner, plan['clients'])
            plan['worker_started'] = True
            os.environ['SUDO_USER'] = plan['user']
            plan['display_stop_attempted'] = True
            save_plan(path, plan)
            runner.run('display-stop', ['/usr/bin/systemctl', 'stop', SERVICE], seconds=30)
            health = script.parent / 'strata-xe-health'
            for method in ['flr', 'bus']:
                with (path.parent / (method + '.log')).open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    code = core.main(['--apply', '--method', method, '--check', str(health)], held_lock=lock, publish=False)
                receipt['attempts'].append(dict(method=method, exit_code=code))
                if code != 1:
                    break  # Success, safety refusal or interruption prevents another reset.
            receipt['recovery_completed'] = code == 0
            if code:
                receipt['error'] = 'Core recovery failed/refused; see attempt logs'
            # Cover GUI startup and the final probe in the same kernel interval.
            cursor = core.journal_cursor(runner, 'before-display-start-cursor')
            restore(core, path, plan, runner, held_lock=lock)
            if code == 0:
                core.check_health(core.Device(BDF), runner, health, cursor=cursor)
                receipt['healthy'] = True
        except Exception as error:
            receipt['error'] = str(error)
            code = 4 if isinstance(error, core.RecoveryRefused) else 1
        finally:
            try:
                restore(core, path, plan, runner, held_lock=lock)
            except Exception as error:
                receipt['display_restore_error'] = str(error)
                receipt['healthy'] = False
                code = 1
            plan = read_plan(path)
            receipt['status'] = 'finished'
            receipt['display_stop_attempted'] = plan['display_stop_attempted']
            receipt['display_restore_requested'] = plan['display_restore_requested']
            receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (path.parent / 'report.json').write_text(json.dumps(receipt, indent=2) + '\n')
            core.publish_summary(receipt, path.parent, runner)
    return code


def main(argv=None):
    script = Path(sys.argv[1]).resolve()
    argv = sys.argv[2:] if argv is None else argv
    if os.geteuid() != 0:
        print('Internal helper requires the sudo-enabled recovery entry.', file=sys.stderr)
        return 4
    core = load_core(script.parent)
    try:
        if not argv:
            return offer(core, script)
        if len(argv) == 2 and argv[0] in ['--worker', '--restore']:
            path = Path(argv[1])
            if argv[0] == '--restore':
                restore_after_exit(core, path)
                return 0
            return worker(core, script, path)
        raise RuntimeError('Invalid internal recovery invocation')
    except Exception as error:
        print('復旧処理を完了できませんでした: ' + str(error), file=sys.stderr)
        return 4


if __name__ == '__main__':
    sys.exit(main())
PY
