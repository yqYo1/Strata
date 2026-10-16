"""Host-native Intel setup adapter for this personal fork.

Reuse Niko's model selection, RAM admission, downloads, packing and server setup.
The HIP installer seam is used only inside this process; written configs use sycl.
No containers, expert-streaming assumptions or fabricated engine versions.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S

INTEL_ARC = {"e223": ("Arc Pro B70", 32), "e221": ("Arc Pro B60", 24),
             "e20b": ("Arc B580", 12), "e20c": ("Arc B570", 10),
             "e212": ("Arc B50", 16), "56a0": ("Arc A770", 16),
             "56a1": ("Arc A750", 8), "56a2": ("Arc A580", 8),
             "56a5": ("Arc A380", 6), "56a6": ("Arc A310", 4)}


def intel_gpus(sysfs=Path("/sys/class/drm")):
    found = []
    for card in sorted(Path(sysfs).glob("card[0-9]*")):
        if "-" in card.name:
            continue
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            devid = (dev / "device").read_text().strip().lower().removeprefix("0x")
            driver = (dev / "driver").resolve().name
        except OSError:
            continue
        if vendor != "0x8086" or driver not in ("xe", "i915") or devid not in INTEL_ARC:
            continue
        name, vram = INTEL_ARC[devid]
        # Prefer a reported physical VRAM size over PCI-id nominal capacity.
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2**30
        except (OSError, ValueError):
            pass
        found.append(dict(index=len(found), name=f"Intel {name}", vram_gb=float(vram),
                          arch="xe", driver=driver))
    return found


def host_path(path):
    return str(Path(path).expanduser().resolve())


def native_paths(environ=None):
    env = os.environ if environ is None else environ
    build = Path(env.get("STRATA_SYCL_BUILD_DIR") or ROOT / "build-sycl").expanduser().resolve()
    exe = Path(env.get("STRATA_SYCL_BIN") or build / S.EXE).expanduser().resolve()
    return build, exe


def build_commands(build, compiler, llama):
    return [["cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
             "-DSTRATA_ENABLE_SYCL=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_ENABLE_HIP=OFF",
             "-DSTRATA_SYCL_TESTS=OFF", "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_CXX_COMPILER={compiler}", f"-DSTRATA_GGML_DIR={llama}"],
            ["cmake", "--build", str(build), "--target", "strata", "-j", str(os.cpu_count() or 2)]]


def cli_flags(source=None):
    text = (ROOT / "src/program/generate.cpp").read_text() if source is None else source
    return set(re.findall(r'\ba\s*==\s*"(--[a-z0-9-]+)"', text))


def validate_config(cfg, supported=None):
    supported = cli_flags() if supported is None else supported
    missing = sorted({str(a).split("=", 1)[0] for a in cfg.get("args", [])
                      if str(a).startswith("--") and str(a).split("=", 1)[0] not in supported})
    if missing:
        raise ValueError("native engine source does not support " + ", ".join(missing)
                         + "; config left unchanged, complete the component port before starting")
    if int(cfg.get("parallel") or 1) > 1 and "BGEN" not in (ROOT / "src/program/generate.cpp").read_text():
        raise ValueError("parallel requests require the pending native BGEN port; config left unchanged")
    if cfg.get("host", "127.0.0.1") not in ("127.0.0.1", "localhost", "::1") and not cfg.get("api_key"):
        raise ValueError("a server beyond localhost requires --api-key")


def to_native(cfg, exe, old=None, vision_explicit=False):
    out = dict(cfg)
    out.update(backend="sycl", exe=host_path(exe), cwd=str(ROOT))
    out["args"] = list(cfg["args"])
    if old:
        # Upstream reports dropped engine flags; on this fork they can be validated
        # tuning choices. Refuse replacement instead of silently losing them.
        dropped = S.args_dropped(old, out)
        if dropped:
            raise ValueError("existing engine options would be lost: " + ", ".join(dropped)
                             + "; edit the config explicitly before rerunning setup")
        if not vision_explicit and old.get("vision"):
            out["vision"] = old["vision"]
        for key in ("host", "api_key", "lib_dirs"):
            if key not in out or (key == "lib_dirs" and not out[key]):
                if key in old:
                    out[key] = old[key]
    validate_config(out)
    return out


def strip_backend(argv):
    rest = []
    it = iter(argv)
    for value in it:
        if value == "--backend":
            next(it, None)
        elif not value.startswith("--backend="):
            rest.append(value)
    return rest


def install(argv):
    argv = strip_backend(argv)
    if S.WIN:
        S.fail("native Intel setup currently requires Linux")
    if any(x.split("=", 1)[0] in ("--cuda", "--gpus", "--layer-split", "--calibrate") for x in argv):
        S.fail("native Intel setup has no CUDA, multi-GPU or automatic calibration adapter",
               "keep the existing config; use the native component validation workflow")
    cards = intel_gpus()
    if not cards:
        S.fail("no supported Intel Arc found in /sys/class/drm", "check the xe/i915 driver and PCI device ID")
    build, exe = native_paths()
    vision_explicit = any(x.split("=", 1)[0] == "--vision" for x in argv)
    forced_build = "--build" in argv

    def engine(gpu, llama, vision="none"):
        if forced_build or not exe.is_file():
            compiler = os.environ.get("CXX") or shutil.which("icpx")
            compiler = shutil.which(compiler) if compiler else None
            if not compiler:
                S.fail("a valid SYCL compiler is required", "set CXX or put icpx on PATH; setup does not alter the environment")
            if exe != build / S.EXE:
                S.fail("--build cannot replace STRATA_SYCL_BIN outside STRATA_SYCL_BUILD_DIR")
            for cmd in build_commands(build, compiler, llama):
                S.run(cmd)
        if not exe.is_file():
            S.fail(f"native engine missing: {exe}")
        if vision != "none" and not (exe.parent / S.VEXE).is_file():
            S.fail("requested native image encoder is missing", f"provide {exe.parent / S.VEXE}; vision choice is preserved")
        # Read the binary's embedded version, never substitute the setup release.
        marker = re.search(rb"engine=(\d+\.\d+\.\d+)\n", exe.read_bytes())
        if marker is None:
            S.fail("native binary has no readable version marker", "rebuild it; setup will not invent a version")
        version = marker.group(1).decode()
        info = exe.parent / "BUILD.json"
        meta = json.loads(info.read_text()) if info.is_file() else {}
        meta.update(version=version, source="local-sycl", backend="sycl", vision=vision)
        meta.setdefault("lib_dirs", [])
        info.write_text(json.dumps(meta, indent=1))
        return exe.parent

    def vision(asked):
        return {"yes": "gpu", "no": "none", None: "none"}.get(asked, asked)

    write_setup = S.write_setup_config
    def write_config(path, cfg, adopted=None):
        previous = path if path.is_file() else adopted
        old = json.loads(previous.read_text(encoding="utf-8-sig")) if previous and previous.is_file() else None
        try:
            cfg = to_native(cfg, exe, old, vision_explicit)
        except ValueError as error:
            S.fail(str(error))
        return write_setup(path, cfg, adopted)

    raw_write = S.write_config
    def checked_write(path, cfg):
        if cfg.get("backend") == "sycl":
            try:
                validate_config(cfg)
            except ValueError as error:
                S.fail(str(error))
        return raw_write(path, cfg)

    upgrade = S.upgrade_config
    def upgrade_config(path, cfg):
        if cfg.get("backend") != "sycl":
            return upgrade(path, cfg)
        try:
            validate_config(cfg)
        except ValueError as error:
            S.fail(str(error))
        return cfg

    installed = S.installed_configs
    def installed_native():
        paths = []
        for path in installed():
            try:
                if json.loads(path.read_text(encoding="utf-8-sig")).get("backend") == "sycl":
                    paths.append(path)
            except (OSError, ValueError):
                continue
        return paths

    original_say = S.say
    S.say = lambda msg="": original_say(str(msg).replace("Your AMD GPUs:", "Your Intel GPUs:")
        .replace("(AMD: docs/AMD_HIP.md)", "(native SYCL)"))
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: cards
    S.amd_problem = lambda g: None
    S.hip_vision = vision
    S.build_engine_hip = engine
    S.cuda_lib_dirs = lambda *a, **k: []
    S.hipblaslt_table = lambda *a, **k: None
    S.write_setup_config = write_config
    S.write_config = checked_write
    S.upgrade_config = upgrade_config
    # Native updates preserve the installed binary; an explicit --build on setup
    # builds it. Never invoke the CUDA release updater on an Intel install.
    S.update_installed_engine = lambda *a, **k: S.say("Native engine retained; use --setup --build to rebuild it.")
    S.setup_calibration = lambda *a, **k: None
    S.installed_configs = installed_native
    sys.argv = [str(ROOT / "setup.py"), *argv, "--backend", "hip"]
    return S.main()


if __name__ == "__main__":
    raise SystemExit(install(sys.argv[1:]))
