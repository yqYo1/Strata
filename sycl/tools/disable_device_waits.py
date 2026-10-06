"""Keep retired GPU-spin kernels out of regenerated SYCL sources."""
import re


def replace_function(source, signature, replacement):
    start = source.find(signature)
    if start < 0:
        return source
    end = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[:start] + replacement + source[end:]


def disable_device_waits(source):
    for name in ('doorbell_wait_kernel', 'wait_flag_ge_kernel', 'wait_flag_ge_or_kernel'):
        source = replace_function(source, '__dpct_inline__ void ' + name + '(', '')
    for name, parameters in (
        ('doorbell_wait', 'const uint32_t*, const uint32_t*, void*'),
        ('wait_flag_ge', 'const uint32_t*, uint32_t, void*'),
        ('wait_flag_ge_or', 'const uint32_t*, uint32_t, const uint32_t*, void*'),
    ):
        source = replace_function(source, 'void ' + name + '(',
            'void ' + name + '(' + parameters + ') {\n'
            '    throw std::logic_error("SYCL ' + name + ' is disabled; use host completion before submission");\n}')
    if 'std::logic_error' in source and '#include <stdexcept>' not in source:
        source = source.replace('#include <cstdlib>', '#include <cstdlib>\n#include <stdexcept>', 1)
    source = source.replace('*(volatile uint32_t*) seq = value;', 'strata::sys_store(seq, value);')
    signature = '__dpct_inline__ void doorbell_publish_value_kernel('
    start = source.find(signature)
    if start >= 0:
        # That publisher writes global/host USM before thread 0 rings the host.
        fragment = replace_function(source[start:], signature, '')
        length = len(source[start:]) - len(fragment)
        publisher = source[start:start+length].replace('fence_space::local_space', 'fence_space::global_and_local')
        source = source[:start] + publisher + source[start+length:]
    return re.sub(r'namespace \{\s*\}  // namespace\s*namespace \{', 'namespace {', source)
