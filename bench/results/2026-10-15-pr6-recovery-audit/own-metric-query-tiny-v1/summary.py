"""Offline summary parser adapted from the pinned project controller b81a06f1.
Repository LICENSE. Complete source delta and origin are in this package.
No timeline, model or RESTORE machinery; keep all original summary/property gates.
"""
import math
import re
import time
from metrics import finite_text, need as require
ERROR = re.compile(r'\[ERROR\]|\[WARNING\]|(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|oracle_error|retirement failed|Strata SYCL cleanup:|Unable to (?:create|trace|initialize)|not enabled for tracing|(?:collection|profile|trace).*truncat', re.I)

def parse_summary(expected, deadline):
    raw = finite_text(expected, deadline)
    require(raw.endswith('\n') and not ERROR.search(raw), 'complete collector log without errors/truncation')
    sections = {'API': [], 'Device': []}
    totals, clocks, headers, properties = {}, {}, {}, []
    section = backend = table = None
    property_header = False
    def name_of(text):
        return text[1:-1] if len(text) >= 2 and text[0] == text[-1] == '"' else text
    for number, line in enumerate(raw.splitlines(), 1):
        require(time.monotonic() < deadline and number <= 10000 and len(line) <= 65536, "finite summary decode")
        match = re.fullmatch(r'=== (API|Device) Timing Summary ===', line.strip())
        if match:
            section, backend, table = match[1], None, None
            require(section not in totals, 'duplicate collector section')
            totals[section] = None
            continue
        if line.strip() == '=== Kernel Properties ===':
            require(section == 'Device' and not property_header, 'single kernel-properties section after device table')
            section, table = 'Properties', None
            continue
        if not line.strip():
            table = None
            continue
        if section == 'Properties':
            fields = [v.strip() for v in line.rsplit(',', 7)]
            head = ['Kernel', 'Compiled', 'SIMD', 'Number of Arguments', 'SLM Per Work Group',
                    'Private Memory Per Thread', 'Spill Memory Per Thread', 'Register File Size Per Thread']
            if fields == head:
                require(not property_header, 'one official kernel-properties header')
                property_header = True
                continue
            require(property_header and len(fields) == 8, 'complete properties row: ' + line)
            name, compiled, simd, nargs, slm, private, spill, grf = fields
            require(compiled in ('AOT', 'JIT') and (simd == 'ANY' or re.fullmatch(r'[1-9][0-9]*', simd)) and
                    all(re.fullmatch(r'[0-9]+', v) for v in (nargs, slm, private, spill)) and
                    (grf == 'unknown' or re.fullmatch(r'[1-9][0-9]*', grf)), 'official kernel property values')
            properties.append(dict(name=name_of(name), compiled=compiled, reported_SIMD=simd,
                arguments=int(nargs), SLM_bytes_per_workgroup=int(slm), private_bytes_per_thread=int(private),
                spill_bytes_per_thread=int(spill), register_file_registers_per_thread=None if grf == 'unknown' else int(grf),
                raw_row=line, scope='Reported compiled-kernel properties; no occupancy or all-input spill guarantee'))
            continue
        if section is None:
            require(not line.strip(), 'unexpected collector preamble')
            continue
        match = re.fullmatch(r'\s*Total Execution Time \(ns\):\s*(\d+)\s*', line)
        if match:
            clocks[section] = int(match[1]); continue
        match = re.fullmatch(r'\s*Total (API|Device) Time for L0 backend \(ns\):\s*(\d+)\s*', line)
        if match:
            require(match[1] == section and totals[section] is None, 'summary type/uniqueness')
            totals[section] = int(match[2]); continue
        if line.strip() == '== L0 Backend ==':
            backend = 'L0'; continue
        fields = [v.strip() for v in line.rsplit(',', 6)]
        head = ['Function' if section == 'API' else 'Kernel', 'Calls', 'Time (ns)', 'Time (%)', 'Average (ns)', 'Min (ns)', 'Max (ns)']
        if fields == head:
            require(backend == 'L0' and section not in headers, 'one L0 table header')
            headers[section] = line; table = section; continue
        require(table == section and backend == 'L0' and len(fields) == 7, 'unparsed/partial timing row: ' + line)
        name, calls, total, percent, average, minimum, maximum = fields
        require(all(re.fullmatch(r'\d+', v) for v in (calls, total, average, minimum, maximum)), 'integer aggregate units')
        calls, total, average, minimum, maximum = map(int, (calls, total, average, minimum, maximum))
        percentage = float(percent)
        require(name and calls > 0 and total >= 0 and math.isfinite(percentage) and 0 <= percentage <= 100.01 and
                0 <= minimum <= maximum and average == total // calls and minimum <= average <= maximum, 'timing row values')
        sections[section].append(dict(name=name_of(name), calls=calls, time_ns=total, percent=percentage,
            average_ns=average, min_ns=minimum, max_ns=maximum, raw_row=line))
    require(set(totals) == set(headers) == set(clocks) == {'API', 'Device'}, 'both emitted summaries required')
    for section, rows in sections.items():
        require(rows and totals[section] and sum(v['time_ns'] for v in rows) == totals[section], 'all rows and exact aggregate sum: ' + section)
    require(any(r['name'].startswith('ze') for r in sections['API']), 'actual Level Zero host API rows')
    require(property_header and properties and all(any(r['name'] == prop['name'] for r in sections['Device']) for prop in properties),
            'compute-kernel properties must match actual device timing rows; copies/barriers alone are not kernel collection')
    result = dict(proved=True,
        headers=headers, totals_ns=totals,
        collector_execution_time_ns=clocks, host_API_rows=sections['API'], device_rows=sections['Device'],
        kernel_properties=properties, command_counts={k: sum(r['calls'] for r in v) for k, v in sections.items()},
        interpretation='Nonexclusive whole-process kernel/command sums and host API elapsed times. Includes startup/teardown and may overlap. Compiled properties are not measured occupancy. No busy union, critical path, prefill-only attribution or clean speed claim.')
    compute = {p["name"] for p in properties}
    result["expected_compute_calls"] = {name: sum(r["calls"] for r in sections["Device"] if r["name"] == name) for name in compute}
    return result

