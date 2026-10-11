# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU-only, single-stage census observer; H2D validation is supplied by caller."""
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


def validate(lines, *, enabled, h2d_validate, h2d_expectations,
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
            if group_calls > a or gi > a or gu > a or gu != down:
                raise ValidationError('wrapper count exceeds active experts or paired dequant differs')
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
