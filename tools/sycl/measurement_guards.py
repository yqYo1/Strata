# Main-authored Strata measurement guards; repository LICENSE applies.
# Extracted unchanged functions from the previous qualified Main controller.
# No device operation, reset, signalling or ownership adoption.
import json
import os
from pathlib import Path
import re
P = Path(__file__).resolve().parents[2]
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
MODEL = re.compile(r'(^|/)(strata(?:[-_.][^/]*)?|sycl_[^/]*|paired_gemm|strata_vllm_xe2_numerics|bench\.sh|serve\.py|engine[-_]wrapper\.py|run_server_job\.py)$')


def require(value, reason):
    if not value:
        raise RuntimeError(reason)

def driver_guard():
    require((DEVICE/'driver').resolve().name == 'xe', 'expected current xe driver')
    require((DEVICE/'vendor').read_text().strip() == '0x8086', 'expected Intel PCI device')
    dumps = list(Path('/sys/class/devcoredump').glob('devcd*'))
    require(not (DEVICE/'devcoredump').exists() and not dumps, 'existing dump; no reset/removal authorized')
    return dict(pci=str(DEVICE), driver='xe', vendor='0x8086', device=(DEVICE/'device').read_text().strip(),
                kernel=Path('/proc/sys/kernel/osrelease').read_text().strip(), devcoredump=False)

def no_model_owners(model_files):
    """Main-time read-only scan; never adopt/signal a discovered owner."""
    observed = []
    population = list(Path('/proc').iterdir())
    require(len(population) < 131072, 'finite proc population')
    for path in population:
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = [a.decode(errors='replace') for a in (path/'cmdline').read_bytes().split(b'\0') if a]
            matches = [a for a in args if not a.startswith('-') and MODEL.search(a)]
            matches += [a for a in args if str(P/'serve') in a and a.endswith('.py')]
            matches += [a for a in args if a in model_files or a.startswith('serve.') or a.endswith('.gguf')]
            if matches:
                observed.append(dict(pid=int(path.name), matching_arguments=matches))
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(not observed, 'model/fixture owners present: ' + json.dumps(observed))
    return dict(same_uid_model_owners=[], scope='bounded read-only cmdline scan; cannot prove hidden foreign GPU use absent')
