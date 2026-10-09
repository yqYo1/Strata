"""Pure frozen-build/current-tree provenance and failed-setup closure rules."""
from pathlib import PurePosixPath
import hashlib
import json
import re

BUILD_COMMIT='1eb89482a4afd20277ae0405780ed4f8eb98eb20'
REVIEWED_HEAD='23268953314d12588fd3f496414a46bd426a7306'
COMPILE_SHA='836656f71cd8dfbf13844ea8f808d2ebc9993f66c65b7548755fb63dee21b8d6'
NINJA_SHA='44d5c41881ea9ea4e5f1c9c38f5be1883c39acd6840071f6914ad7a45c769274'
CODE_SUFFIXES=('.c','.cc','.cpp','.cxx','.h','.hh','.hpp','.hxx','.cuh','.inc','.in','.cmake')


def require(ok,message):
    if not ok: raise ValueError(message)


def relative_path(path):
    require(type(path) is str and path and not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts and str(PurePosixPath(path))==path,'canonical relative source path')
    return path


def conservative_inputs(tracked,compiled):
    """Explicit compile TUs plus conservative tracked code/headers/CMake superset.

    Includes even unused project code/headers, rejecting rather than guessing
    whether a modified include was consumed. Not a system-header provenance oracle.
    """
    require(type(tracked) is list and len(tracked)<=20000 and len(set(tracked))==len(tracked),'tracked geometry/duplicates')
    require(type(compiled) is list and 1<=len(compiled)<=512,'compiled geometry')
    paths={relative_path(p) for p in compiled}
    for p in tracked:
        relative_path(p)
        if p.endswith(CODE_SUFFIXES) or PurePosixPath(p).name=='CMakeLists.txt': paths.add(p)
    require(paths<=set(tracked),'compiled source absent from build commit')
    return sorted(paths)


def compile_paths(commands,root,ggml_root,build_dir):
    require(type(commands) is list and 1<=len(commands)<=512,'compile command geometry')
    result={'project':set(),'ggml':set()}
    roots={'project':PurePosixPath(root),'ggml':PurePosixPath(ggml_root)}
    for command in commands:
        require(type(command) is dict and command.get('directory')==build_dir and type(command.get('command')) is str and command['command'],'compile directory/command')
        path=command.get('file');require(type(path) is str and PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts,'absolute compile path')
        matches=[(key,parent) for key,parent in roots.items() if PurePosixPath(path).is_relative_to(parent)]
        require(len(matches)==1,'compile file outside pinned roots')
        key,parent=matches[0];result[key].add(relative_path(str(PurePosixPath(path).relative_to(parent))))
    require('src/kernels/cpu/native_expert.cpp' in result['project'] and 'sycl/src/kernels/cpu/native_expert.cpp' not in result['project'],'actual production native source path')
    return {key:sorted(paths) for key,paths in result.items()}


def validate_closure(build_commit,current_head,status,expected,current,changed):
    require(build_commit==BUILD_COMMIT and current_head==REVIEWED_HEAD,'exact reviewed build/current HEAD')
    require(status=='','dirty baseline tree')
    require(type(expected) is dict and expected and type(current) is dict and set(expected)==set(current),'exact closure keys')
    for path,pin in expected.items():
        relative_path(path)
        require(type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin),'expected source SHA256')
        require(current[path]==pin,'changed compiled-input source: '+path)
    require(type(changed) is list and len(set(changed))==len(changed),'changed-file geometry')
    for p in changed: relative_path(p)
    require(not set(changed)&set(expected),'build-to-current changed closure path')
    canonical=(json.dumps(expected,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')
    return dict(build_commit=build_commit,current_clean_HEAD=current_head,compiled_input_count=len(expected),compiled_input_sha256=hashlib.sha256(canonical).hexdigest(),changed_noncompiled_files=sorted(changed),closure_policy='exact compile TUs plus conservative tracked code/header/CMake superset; unchanged hashes required')


def failed_setup(error,cleanup=None):
    """A setup failure cannot upgrade any numerical or lifecycle gate."""
    return dict(active=False,completed=False,healthy=False,math_gate_passed=False,reference_established=False,protocol_boundary_gate_passed=False,text_budget_gate_passed=False,error=repr(error)[:2048],cleanup=cleanup if cleanup is not None else {'forced':False,'inferior_survived':False,'gdb_survived':False},performance_eligible=False,adopted=False,full_lifecycle_passed=False)


def close_setup_resources(resources):
    """Close every acquired stream even when an earlier close raises."""
    errors=[]
    for resource in resources:
        if resource is None: continue
        try: resource.close()
        except BaseException as error: errors.append(repr(error)[:2048])
    return errors
