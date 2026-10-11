# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ON-only observer; independent model exclusions and H2D validation required."""
import re
from collections.abc import Mapping

U64 = (1 << 64) - 1
I64 = (1 << 63) - 1
FAMILY = 'strata prefill route census'
KEYS = ('device', 'lb', 'le', 'offset', 'pos', 'tokens', 'layer')
FLAGS = ('routing_observed', 'mmq_enabled', 'plan_fallback', 'mmq_direct', 'group_gather')
COUNTERS = ('routed_rows', 'expected_rows', 'active_experts', 'active_nonresident',
            'zero_nonresident', 'gather_group_accepted_calls',
            'gather_group_declined_calls', 'gather_individual_calls',
            'native_dequant_gu_calls', 'native_dequant_down_calls')
D = r'(?:0|[1-9][0-9]*)'
ROW = re.compile(
    rf'strata prefill route census v1: device=(?P<device>{D}) '
    rf'stage=\[(?P<lb>{D}),(?P<le>{D})\) chunk_offset=(?P<offset>{D}) '
    rf'pos=(?P<pos>{D}) tokens=(?P<tokens>{D}) layer=(?P<layer>{D}) '
    r'mode=(?P<mode>full_nonresident_stream|routed) issuer=(?P<issuer>thread|inline) '
    + ' '.join(rf'{key}=(?P<{key}>[01])' for key in FLAGS) + ' '
    + ' '.join(rf'{key}=(?P<{key}>{D})' for key in COUNTERS)
    + ' completion=checked')


class ValidationError(ValueError):
    """Invalid explicit scope, census, or joined ledger."""


def integer(value, name, maximum=U64, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f'{name}: integer outside [{minimum},{maximum}]')
    return value


def number(token, name, maximum):
    if len(token) > 20:
        raise ValidationError(f'{name}: decimal overflow')
    return integer(int(token), name, maximum)


def capture_lines(lines):
    if isinstance(lines, str):
        lines = lines.split('\n')
    result = []
    for raw in lines:
        if not isinstance(raw, str):
            raise ValidationError('capture line must be text')
        line = raw.removesuffix('\n').removesuffix('\r')
        if '\n' in line or '\r' in line:
            raise ValidationError('embedded line boundary')
        result.append(line)
    return result


def _validate_routes(lines, *, enabled, h2d_validate, h2d_expectations,
             n_experts, route_k, moe_layers, nonresident_population=None):
    """Validate complete same-capture ledger, then join exact census rows.

    h2d_validate must be the independently qualified transfer_ledger.validate;
    h2d_expectations is its explicit geometry/policy keyword mapping. No receipt
    supplied by an untrusted capture is accepted as independent validation.
    Model dimensions and MoE IDs come from the pinned model, never the capture.
    Disabled mode requires absence of both census and transfer-ledger families.
    """
    if type(enabled) is not bool:
        raise ValidationError('enabled must be an explicit bool')
    text = capture_lines(lines)
    if not enabled:
        if any(FAMILY in line or 'strata prefill expert' in line for line in text):
            raise ValidationError('disabled diagnostics present')
        return {'scope': 'this_prefill_stage_run', 'enabled': False,
                'absent': True, 'independent_completion_validation': False}
    integer(n_experts, 'model n_experts', (1 << 31) - 1, 1)
    integer(route_k, 'model route K', n_experts, 1)
    if not isinstance(moe_layers, (set, frozenset)):
        raise ValidationError('moe_layers must be an explicit set')
    for layer in moe_layers:
        integer(layer, 'MoE layer', I64)
    if not isinstance(h2d_expectations, Mapping) or not callable(h2d_validate):
        raise ValidationError('require qualified H2D validator and expectations')
    ledger = h2d_validate(text, **h2d_expectations)
    expected = ledger['rows']
    stage_layers = {row['layer'] for row in expected}
    if nonresident_population is not None:
        if not isinstance(nonresident_population, Mapping) or set(nonresident_population) != stage_layers & moe_layers:
            raise ValidationError('population map must cover stage MoE layers exactly')
        for layer, population in nonresident_population.items():
            integer(layer, 'population layer', I64)
            integer(population, 'nonresident population', n_experts)
    rows = []
    totals = {key: 0 for key in COUNTERS}
    h2d_count = 0
    total_seen = False
    plan = None
    layer_plan = {}
    for line in text:
        if line.startswith('strata prefill expert H2D:'):
            if rows or total_seen:
                raise ValidationError('H2D row after census or total')
            h2d_count += 1
        if line.startswith('strata prefill expert H2D total:'):
            if total_seen or len(rows) != len(expected):
                raise ValidationError('total before census coverage or duplicate total')
            total_seen = True
        if FAMILY not in line:
            continue
        match = ROW.fullmatch(line) if len(line) <= 2048 else None
        if match is None:
            raise ValidationError('malformed recognizable census family')
        if total_seen or h2d_count != len(expected) or len(rows) >= len(expected):
            raise ValidationError('extra census or wrong emission order')
        row = match.groupdict()
        for key in KEYS + FLAGS + COUNTERS:
            row[key] = number(row[key], key, ((1 << 31) - 1) if key == 'device' else I64 if key in KEYS else U64)
        other = expected[len(rows)]
        if any(row[key] != other[key] for key in KEYS + ('mode', 'issuer')):
            raise ValidationError('wrong scope, identity, policy or census order')
        if plan is not None and row['plan_fallback'] != plan:
            raise ValidationError('global plan_fallback changed')
        plan = row['plan_fallback']
        layer = row['layer']
        if layer in layer_plan and row['mmq_enabled'] != layer_plan[layer]:
            raise ValidationError('per-layer MMQ plan changed')
        layer_plan[layer] = row['mmq_enabled']
        if row['routing_observed'] != int(layer in moe_layers):
            raise ValidationError('routing observation disagrees with model MoE IDs')
        if not row['routing_observed']:
            if row['mmq_direct'] or row['group_gather'] or any(row[key] for key in COUNTERS):
                raise ValidationError('unobserved routing has counters or execution policy')
        else:
            product = integer(row['tokens'] * route_k, 'tokens*K')
            if row['routed_rows'] != product or row['expected_rows'] != product:
                raise ValidationError('actual and expected routed rows differ from tokens*K')
            a, b, z = (row[key] for key in ('active_experts', 'active_nonresident', 'zero_nonresident'))
            population = integer(b + z, 'nonresident count sum')
            if not b <= a <= min(n_experts, product) or population > n_experts or a - b > n_experts - population:
                raise ValidationError('inconsistent active/nonresident/zero counts')
            if product and not a:
                raise ValidationError('positive routed rows without active experts')
            if nonresident_population is not None and population != nonresident_population[layer]:
                raise ValidationError('external nonresident population mismatch')
            direct, group, mmq = (row[key] for key in ('mmq_direct', 'group_gather', 'mmq_enabled'))
            ga, gd, gi, gu, down = (row[key] for key in COUNTERS[5:])
            group_calls = integer(ga + gd, 'group calls')
            if direct and group or (direct or group) and not mmq or group and row['mode'] != 'full_nonresident_stream':
                raise ValidationError('inconsistent direct/group policy')
            if group_calls and not group or direct and (group_calls or gi) or gd > gi:
                raise ValidationError('inconsistent group/individual wrappers')
            if group_calls > a or gi > a or gu > a :
                raise ValidationError('wrapper count exceeds active experts ')
            if mmq and (gu or down) or not mmq and (group_calls or gi):
                raise ValidationError('wrapper conflicts with MMQ plan')
            if other['format'] != 'native_gguf' and (mmq or group_calls or gi or gu or down):
                raise ValidationError('native-only wrappers/plan on canonical format')
        for key in COUNTERS:
            totals[key] = integer(totals[key] + row[key], f'{key} aggregate')
        rows.append(row)
    if not total_seen or len(rows) != len(expected):
        raise ValidationError('missing completed census coverage')
    return {'scope': 'this_prefill_stage_run', 'enabled': True, 'capture_valid': True,
            'rows': rows, 'aggregates': totals,
            'aggregate_units': {key: 'host wrapper normal returns' if key.endswith('_calls') else
                                'layer/chunk routed row occurrences' if key.endswith('_rows') else
                                'layer/chunk expert occurrences' for key in COUNTERS},
            'h2d_total': ledger['total'], 'completion_declaration': 'checked',
            'independent_completion_validation': False, 'gpu_execution_inferred': False}


GROUP_FAMILY = '[prefill-native-dequant-group-v2]'
OLD_GROUP_FAMILY = '[prefill-native-dequant-group-v1]'
GROUP_CONTEXT = ('device', 'lb', 'le', 'chunk_offset', 'pos', 'tokens', 'layer',
                 'fused_prefix', 'cpu_experts', 'gpu_order')
GROUP_FIELDS = GROUP_CONTEXT + ('gu_admitted', 'gu_declined', 'down_admitted',
                'down_declined', 'gu_covered', 'down_covered',
                'gu_fallback_experts', 'down_fallback_experts', 'flushes',
                'held_reuse_flushes', 'released_slots')
GROUP_ROW = re.compile(re.escape(GROUP_FAMILY) + ' ' + ' '.join(
    rf'{key}=(?P<{key}>{D})' for key in GROUP_FIELDS))


def validate(lines, *, enabled, h2d_validate, h2d_expectations,
             n_experts, route_k, moe_layers, row_expectations,
             nonresident_population=None):
    """Strict ON capture validation for exactly one completed stage/run.

    row_expectations must independently name every native !MMQ MoE scope as
    (chunk_offset, layer): {fused_prefix, cpu_experts, selected_streamed_experts}.
    These counts must come from independently pinned selection evidence, never
    from this diagnostic. Zero exclusions require explicit zero pins. The source
    v2 grammar supplies full context: rows join ordered exact eligible scope
    and must all precede the end-of-run H2D ledger. Multi-run/interleaved
    captures are unsupported. Original OFF validator remains a separate unmodified file.
    h2d_validate is the independently qualified original transfer_ledger.validate.
    """
    text = capture_lines(lines)
    if any(OLD_GROUP_FAMILY in line for line in text):
        raise ValidationError('v1 group diagnostics unsupported in v2 capture')
    if type(enabled) is not bool:
        raise ValidationError('enabled must be explicit bool')
    if not enabled:
        if any(GROUP_FAMILY in line for line in text):
            raise ValidationError('disabled group diagnostics present')
        return _validate_routes(text, enabled=False, h2d_validate=h2d_validate,
                                h2d_expectations=h2d_expectations,
                                n_experts=n_experts, route_k=route_k,
                                moe_layers=moe_layers)
    if not isinstance(row_expectations, Mapping):
        raise ValidationError('independent exact row_expectations required')
    for scope in row_expectations:
        if type(scope) is not tuple or len(scope) != 2:
            raise ValidationError('row expectation key must be exact offset/layer tuple')
        for value in scope:
            integer(value, 'independent scope key', I64)
    result = _validate_routes(text, enabled=True, h2d_validate=h2d_validate,
                              h2d_expectations=h2d_expectations,
                              n_experts=n_experts, route_k=route_k,
                              moe_layers=moe_layers,
                              nonresident_population=nonresident_population)
    ledger = h2d_validate(text, **h2d_expectations)
    eligible = []
    for route, transfer in zip(result['rows'], ledger['rows']):
        if (route['routing_observed'] and not route['mmq_enabled']
                and transfer['format'] == 'native_gguf'):
            eligible.append((route, transfer))
        elif route['native_dequant_gu_calls'] != route['native_dequant_down_calls']:
            raise ValidationError('unpaired wrappers outside eligible ON scope')
    scopes = {(r['offset'], r['layer']) for r, _ in eligible}
    if set(row_expectations) != scopes:
        raise ValidationError('independent pins must cover eligible scopes exactly')
    groups = []
    ledger_started = False
    for line in text:
        if 'strata prefill expert' in line or FAMILY in line:
            ledger_started = True
        if GROUP_FAMILY not in line:
            continue
        match = GROUP_ROW.fullmatch(line) if len(line) <= 2048 else None
        if match is None or ledger_started or len(groups) >= len(eligible):
            raise ValidationError('malformed/extra/late group row or ambiguous scope')
        group = {key: number(value, key, ((1 << 31) - 1) if key == 'device'
                             else I64 if key in GROUP_CONTEXT and key not in
                             ('fused_prefix', 'gpu_order') else U64)
                 for key, value in match.groupdict().items()}
        route, transfer = eligible[len(groups)]
        for key in KEYS:
            group_key = 'chunk_offset' if key == 'offset' else key
            if group[group_key] != route[key]:
                raise ValidationError('group context differs from ordered route/H2D scope')
        pin = row_expectations[(route['offset'], route['layer'])]
        fields = {'fused_prefix', 'cpu_experts', 'selected_streamed_experts'}
        if not isinstance(pin, Mapping) or set(pin) != fields:
            raise ValidationError('require exact independent exclusion/stream pins')
        for field in fields:
            integer(pin[field], field, n_experts)
        active = route['active_experts']
        nonresident = route['active_nonresident']
        fused, cpu, selected = (pin[key] for key in
                               ('fused_prefix', 'cpu_experts', 'selected_streamed_experts'))
        if fused > active - nonresident or cpu > nonresident or fused + cpu > active:
            raise ValidationError('impossible fused/CPU exclusions')
        if group['fused_prefix'] != fused or group['cpu_experts'] != cpu:
            raise ValidationError('observed fused/CPU exclusions differ from independent pins')
        if group['gpu_order'] != active - cpu or group['gpu_order'] < fused:
            raise ValidationError('observed GPU order inconsistent with routing/exclusions')
        logical = group['gpu_order'] - group['fused_prefix']
        if selected != nonresident - cpu or selected > logical:
            raise ValidationError('selected streamed experts disagree with route exclusions')
        if transfer['mode'] == 'routed' and transfer['calls'] != selected:
            raise ValidationError('routed H2D submissions disagree with selected stream')
        if transfer['mode'] == 'full_nonresident_stream' and transfer['calls'] != nonresident + route['zero_nonresident']:
            raise ValidationError('full-stream H2D disagrees with route population')
        flushes = group['flushes']
        if not (logical == 0 and flushes == 0 or 1 <= flushes <= logical <= 16 * flushes):
            raise ValidationError('flush windows outside 1..16 or nonzero empty scope')
        for axis in ('gu', 'down'):
            admitted, declined, covered, fallback = (group[axis + suffix] for suffix in
                ('_admitted', '_declined', '_covered', '_fallback_experts'))
            if admitted + declined != flushes or covered + fallback != logical:
                raise ValidationError('axis flush/coverage sum mismatch')
            for count, experts in ((admitted, covered), (declined, fallback)):
                if not (count == experts == 0 or 1 <= count <= experts <= 16 * count):
                    raise ValidationError('axis window coverage bounds violated')
            if route['native_dequant_' + axis + '_calls'] != fallback:
                raise ValidationError('old wrapper returns differ from own fallback experts')
        if group['held_reuse_flushes'] > flushes or group['released_slots'] != selected:
            raise ValidationError('reuse/released slots disagree with completed selected stream')
        if transfer['mode'] != 'routed' and group['held_reuse_flushes']:
            raise ValidationError('held stage reuse outside routed scope')
        groups.append(dict(group, offset=route['offset'], logical_experts=logical))
    if len(groups) != len(eligible):
        raise ValidationError('missing group row; missing is never completion')
    result['group_rows'] = groups
    result['group_units'] = 'host normal-return admissions and logical expert coverage'
    result['window_validation'] = 'aggregate bounds only; individual windows are not logged'
    result['gpu_execution_inferred'] = False
    result['performance_inferred'] = False
    return result
