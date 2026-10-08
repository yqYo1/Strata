"""Defer migration-created codebooks until kernels actually request them."""
import re


def lazy_device_tables(source):
    for name, helper in [('iq4nl_values', 'iq4nl_storage'), ('c_codes', 's2_codes_storage')]:
        pattern = (r'inline (dpct::(?:global|constant)_memory<[^>]+>)& ' + name +
                   r' = \*(new dpct::[^;]+);[^\n]*')
        source = re.sub(pattern, lambda m: 'inline ' + m[1] + '& ' + helper + '() {\n'
                        '    static auto* table = ' + m[2] + ';\n    return *table;\n}', source)
        source = source.replace(name + '.', helper + '().')
    return source.replace('    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);',
                          '    uint32_t table32[4];\n    __builtin_memcpy(table32, iq4nl_values, sizeof table32);')
