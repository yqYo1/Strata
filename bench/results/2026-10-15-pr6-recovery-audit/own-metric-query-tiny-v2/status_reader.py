"""Strict offline reader for the separately pinned Linux tiny query collector.
Repository LICENSE. Pure standard library; no device, imports of runtime, or exec.
"""
import json
from pathlib import Path
import time

PREFIX = 'OWN_METRIC_V2 '
APIS = {
    'context-create':'zeContextCreate', 'context-destroy':'zeContextDestroy',
    'group-count':'zetMetricGroupGet', 'group-list':'zetMetricGroupGet',
    'group-properties':'zetMetricGroupGetProperties', 'schema-group':'zetMetricGroupGetProperties',
    'activate':'zetContextActivateMetricGroups', 'deactivate':'zetContextActivateMetricGroups',
    'pool-create':'zetMetricQueryPoolCreate', 'pool-destroy':'zetMetricQueryPoolDestroy',
    'query-create':'zetMetricQueryCreate', 'query-destroy':'zetMetricQueryDestroy',
    'query-begin':'zetCommandListAppendMetricQueryBegin', 'query-end':'zetCommandListAppendMetricQueryEnd',
    'query-ready':'zeEventQueryStatus', 'data-size':'zetMetricQueryGetData',
    'data-read':'zetMetricQueryGetData', 'query-reset':'zetMetricQueryReset',
    'schema-list':'zetMetricGet', 'schema-metric':'zetMetricGetProperties',
    'calculate-size':'zetMetricGroupCalculateMultipleMetricValuesExp',
    'calculate-read':'zetMetricGroupCalculateMultipleMetricValuesExp',
}

FIELDS = {
    'context-create': ('', 'context'), 'context-destroy': ('context', ''),
    'group-count': ('device count', 'count'), 'group-list': ('device count', 'count'),
    'group-properties': ('group index', 'count flags domain group_name'),
    'schema-group': ('group', 'count flags domain group_name'),
    'activate': ('context device group count', ''), 'deactivate': ('context device count', ''),
    'pool-create': ('context device group count type', 'pool'), 'pool-destroy': ('pool', ''),
    'query-create': ('pool index', 'query'), 'query-destroy': ('query', ''),
    'query-begin': ('list query', ''), 'query-end': ('list query event', ''),
    'query-ready': ('query instance event', ''), 'data-size': ('query instance size', 'size'),
    'data-read': ('query instance size', 'size'), 'query-reset': ('query', ''),
    'schema-list': ('group count', 'count'),
    'schema-metric': ('metric', 'metric_name units result_type'),
    'calculate-size': ('group instance size samples values', 'samples values'),
    'calculate-read': ('group instance size samples values', 'samples values sample0'),
}

def data_schema(data, site, output=False):
    keys = set(FIELDS[site][int(output)].split())
    need(type(data) is dict and set(data) == keys | {'present'} and data['present'] is True,
         'exact operation metadata schema: ' + site)
    for key in keys:
        value = data[key]
        if key in ('group_name','metric_name','units'):
            need(type(value) is str and len(value.encode()) <= 1024 and '\x00' not in value,
                 'bounded metadata text')
        else:
            need(integer(value), 'exact unsigned metadata integer')


def need(value, message):
    if not value:
        raise ValueError('HOLD_API_EVIDENCE: ' + message)

def integer(value, low=0, high=2**64-1):
    return type(value) is int and low <= value <= high

def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, 'duplicate JSON key')
        out[key] = value
    return out

def reconcile(text, pid, query, catalog_group, catalog, deadline, codes):
    """Every entry/result plus complete query/sample/CSV/summary reconciliation.

    `query` comes only from the unchanged strict v1 metric decoder, whose rows
    already reconcile exact per-kernel compute counts with both summary tables.
    Handles are process-local IDs, not a device identity claim on their own.
    """
    need(integer(pid, 1) and text.endswith('\n') and '\x00' not in text and len(text.encode()) <= 62 << 20,
         'finite complete native stderr')
    records = []
    for line in text.splitlines():
        need(time.monotonic() < deadline, 'reader deadline')
        if PREFIX not in line:
            continue
        need(line.startswith(PREFIX) and len(line.encode()) <= 4096, 'interleaved/oversized audit line')
        row = json.loads(line[len(PREFIX):], object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        need(type(row) is dict and type(row.get('schema')) is int and row.get('schema') == 2 and
             integer(row.get('pid'),1) and row.get('pid') == pid and
             integer(row.get('seq'), 1) and row['seq'] == len(records)+1, 'PID/schema/sequence gap')
        records.append(row)
        need(len(records) <= 100000, 'finite status-record budget')
    need(len(records) >= 4 and records[0]['phase'] == 'open' and
         records[0].get('route') == 'Linux_EVENT_BASED_ComputeBasic' and records[-1]['phase'] == 'close',
         'open/close; empty or truncated stream')
    need(query['decoded'] and len(catalog) == 83 and catalog_group['target'] and
         catalog_group['name'] == 'ComputeBasic' and catalog_group['sampling_flags'] == 1,
         'Main full catalog and original strict CSV decoder')
    common_keys = {'schema','pid','seq','phase'}
    need(set(records[0]) == common_keys | {'route'} and
         set(records[-1]) == common_keys | {'calls'} and integer(records[-1]['calls'],1), 'exact lifecycle schema')
    calls, completed, submissions, collected, pending = {}, [], {}, {}, {}
    result_ids = set()
    for row in records[1:-1]:
        need(time.monotonic() < deadline, 'reconciliation deadline')
        phase = row.get('phase')
        need(phase in ('entry','result','submit','collected'), 'rejected/unavailable/unknown/duplicate lifecycle')
        if phase in ('submit','collected'):
            need(set(row) == common_keys | {'query','instance'}, 'exact sample-binding schema')
            q, instance = row.get('query'), row.get('instance')
            need(integer(q, 1) and integer(instance, 1), 'query/sample binding')
            if phase == 'submit':
                need(instance not in submissions and q not in pending, 'duplicate/overlapping submission')
                submissions[instance] = (q, row['seq']); pending[q] = instance
            else:
                need(pending.get(q) == instance and instance not in collected, 'unmatched collection completion')
                collected[instance] = (q, row['seq']); del pending[q]
            continue
        call_id, site = row.get('call'), row.get('site')
        need(integer(call_id, 1) and site in APIS and row.get('api') == APIS[site], 'exact call ID/site/API')
        if phase == 'entry':
            need(set(row) == common_keys | {'call','site','api','data'}, 'exact entry schema')
            data_schema(row.get('data'), site)
            need(call_id == len(calls)+1 and call_id not in calls and type(row.get('data')) is dict and
                 row['data'].get('present') is True, 'entry order/data')
            calls[call_id] = row
        else:
            need(set(row) == common_keys | {'call','site','api','code','hex','name','data'}, 'exact result schema')
            need(call_id in calls and calls[call_id]['site'] == site and
                 calls[call_id]['api'] == row['api'] and call_id not in result_ids,
                 'missing/duplicate/mismatched result')
            code = row.get('code'); name = row.get('name')
            need(integer(code, 0, 2**32-1) and name in codes and codes[name] == code and
                 row.get('hex') == hex(code), 'exact numeric/hex/symbolic result')
            need(code == 0 or (site == 'query-ready' and code == 1 and name == 'ZE_RESULT_NOT_READY'),
                 'failed/unsupported/unavailable metric API')
            need((code == 0 and type(row.get('data')) is dict and row['data'].get('present') is True) or
                 (code == 1 and row.get('data') is None), 'successful output evidence')
            if code == 0: data_schema(row['data'], site, True)
            completed.append((calls[call_id], row)); result_ids.add(call_id)
    need(not pending and len(completed) == len(calls) == records[-1].get('calls'), 'all calls/results/owners closed')
    by = {site:[] for site in APIS}
    for entry, result in completed:
        by[entry['site']].append((entry['data'], result['data'], result['code'], entry['seq'], result['seq']))
    def one(site):
        need(len(by[site]) == 1, 'exactly one ' + site)
        return by[site][0]
    def handle(value):
        need(integer(value, 1), 'nonzero process-local handle'); return value
    create = one('context-create'); context = handle(create[1]['context'])
    count = one('group-count'); listing = one('group-list')
    device = handle(count[0]['device'])
    need(count[0]['count'] == 0 and integer(count[1]['count'], 1, 256) and
         listing[0]['device'] == device and listing[0]['count'] == listing[1]['count'] == count[1]['count'] and
         create[4] < count[3] < count[4] < listing[3] < listing[4], 'exact bounded discovery')
    activation = one('activate'); a = activation[0]; group = handle(a['group'])
    need(a['context'] == context and a['device'] == device and a['count'] == 1, 'one selected activation')
    properties = by['group-properties']
    need(properties and len(properties) <= count[1]['count'] and
         len({p[0]['group'] for p in properties}) == len(properties) and
         [p[0]['index'] for p in properties] == list(range(catalog_group['group']+1)),
         'bounded unique discovered properties and exact selected catalog index')
    chosen = [p for p in properties if p[0]['group'] == group]
    need(len(chosen) == 1 and chosen[0] == properties[-1] and
         all(listing[4] < p[3] < p[4] < activation[3] for p in properties), 'selected group exact discovery path')
    for props in (chosen[0][1], one('schema-group')[1]):
        need((props['group_name'],props['flags'],props['count'],props['domain']) ==
             ('ComputeBasic',1,83,catalog_group['domain']), 'selected EVENT_BASED full catalog metadata')
    sg = one('schema-group'); sl = one('schema-list')
    need(sg[0]['group'] == sl[0]['group'] == group and sl[0]['count'] == sl[1]['count'] == 83 and
         sg[4] < sl[3], 'exact calculation schema calls')
    schema = by['schema-metric']
    need(len(schema) == 83 and len({p[0]['metric'] for p in schema}) == 83 and
         all(sl[4] < p[3] for p in schema), 'all 83 metric property statuses')
    need([(p[1]['metric_name'],p[1]['units'],p[1]['result_type']) for p in schema] ==
         [(m['name'],m['units'],m['result_type']) for m in catalog], 'ordered exact actual catalog')
    pools = {}; queries = {}
    for pre,post,_,start,end in by['pool-create']:
        pool = handle(post['pool'])
        need(pool not in pools and pre['device'] == device and pre['group'] == group and
             handle(pre['context']) and pre['count'] == 64 and pre['type'] == 0 and activation[4] < start,
             'original performance query pool')
        pools[pool] = (start,end)
    need(pools and len(pools) <= 64, 'bounded nonempty pools')
    for pre,post,_,start,end in by['query-create']:
        q = handle(post['query']); pool = pre['pool']
        need(q not in queries and pool in pools and integer(pre['index'], 0, 63) and pools[pool][1] < start,
             'unique successful query creation')
        queries[q] = (pool,pre['index'],start,end)
    for pool in pools:
        need(sorted(v[1] for v in queries.values() if v[0] == pool) == list(range(64)), 'all original 64 queries created')
    for site in ('query-begin','query-end','query-reset','query-destroy'):
        need(all(p[0]['query'] in queries and queries[p[0]['query']][3] < p[3] for p in by[site]), 'created query at ' + site)
    # One begin/end per appended kernel. Non-immediate resubmission may use the
    # same recorded pair, but every actual sample still needs distinct statuses.
    begins = {}; ends = {}
    for pre,_,_,start,end in by['query-begin']:
        begins.setdefault(pre['query'], []).append((pre['list'],start,end))
    for pre,_,_,start,end in by['query-end']:
        ends.setdefault(pre['query'], []).append((pre['list'],pre['event'],start,end))
    need(set(begins) == set(ends) and begins, 'all appended query ends reached')
    for q in begins:
        need(len(begins[q]) == len(ends[q]) and all(b[0] == e[0] and b[2] < e[2]
             for b,e in zip(begins[q],ends[q])), 'matching original begin/end/list order')
    csv_ids = {r['global_instance_id'] for r in query['rows']}
    need(csv_ids and set(submissions) == set(collected) == csv_ids and len(csv_ids) <= 4096,
         'exact CSV/summary/submission/collected instance set')
    need(len(by['query-reset']) == len(csv_ids), 'one exact successful reset per sample')
    used_resets = set(); sizes = {}
    for instance in sorted(csv_ids):
        need(time.monotonic() < deadline, 'sample reconciliation deadline')
        q,submitted_seq = submissions[instance]; _,collected_seq = collected[instance]
        need(q in begins and any(e[3] < submitted_seq for e in ends[q]), 'query end before actual submit')
        def sample(site):
            rows = [p for p in by[site] if p[0].get('instance') == instance]
            need(len(rows) == 1 and rows[0][0]['query'] == q, 'one exact ' + site + ' per sample')
            return rows[0]
        ready = [p for p in by['query-ready'] if p[0]['instance'] == instance]
        success = [p for p in ready if p[2] == 0]
        size = sample('data-size'); data = sample('data-read')
        need(success and all(p[0]['query'] == q and submitted_seq < p[3] < p[4] < size[3] for p in ready) and
             all(any(p[0]['event'] == e[1] for e in ends[q]) for p in ready), 'explicit event readiness before read')
        need(size[0]['size'] == 0 and integer(size[1]['size'],1,62 << 20) and
             size[1]['size'] == data[0]['size'] == data[1]['size'] and size[4] < data[3], 'both exact GetData sizes/statuses')
        resets = [p for p in by['query-reset'] if p[0]['query'] == q and data[4] < p[3] < p[4] < collected_seq]
        need(len(resets) == 1 and resets[0][3] not in used_resets, 'reset after data and before collected')
        used_resets.add(resets[0][3]); sizes[instance] = size[1]['size']
    for site in ('query-ready','data-size','data-read'):
        need(all(p[0]['instance'] in csv_ids for p in by[site]), 'no unrepresented ' + site)
    for site in ('calculate-size','calculate-read'):
        need(len(by[site]) == len(csv_ids) and {p[0]['instance'] for p in by[site]} == csv_ids, 'exact ' + site + ' coverage')
    for instance in csv_ids:
        need(time.monotonic() < deadline, 'calculation reconciliation deadline')
        calc = [next(p for p in by[site] if p[0]['instance'] == instance) for site in ('calculate-size','calculate-read')]
        first,second = calc
        need(first[0]['samples'] == first[0]['values'] == 0 and
             second[0]['samples'] == first[1]['samples'] == second[1]['samples'] == 1 and
             second[0]['values'] == first[1]['values'] == second[1]['values'] == second[1]['sample0'] == 83 and
             all(p[0]['group'] == group and p[0]['size'] == sizes[instance] for p in calc) and
             collected[instance][1] < first[3] < first[4] < second[3] and
             all(p[4] < first[3] for p in schema), 'exact raw-byte/single-set/83-value calculation')
    deactivation = one('deactivate'); destruction = one('context-destroy')
    need(deactivation[0] == dict(present=True,context=context,device=device,count=0) and
         destruction[0] == dict(present=True,context=context) and
         all(p[4] < deactivation[3] for p in by['calculate-read']) and deactivation[4] < destruction[3], 'activation context closure')
    destroyed_q = by['query-destroy']; destroyed_p = by['pool-destroy']
    need(len(destroyed_q) == len(queries) and {p[0]['query'] for p in destroyed_q} == set(queries) and
         len(destroyed_p) == len(pools) and {p[0]['pool'] for p in destroyed_p} == set(pools), 'all original query/pool destruction results')
    need(all(destruction[4] < p[3] for p in destroyed_q) and
         all(max(p[4] for p in destroyed_q) < p[3] for p in destroyed_p), 'original teardown order observed')
    need(time.monotonic() < deadline, 'complete finite reconciliation')
    return dict(proved=True, exact_api_statuses=True, calls=len(calls), records=len(records),
                samples=len(csv_ids), instances=sorted(csv_ids), pending_polls=len([p for p in by['query-ready'] if p[2] == 1]),
                limitation='Selected PTI metric call sites only; no provider internals, model or performance qualification')

def read(path, pid, query, catalog_group, catalog, deadline):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and 0 < p.stat().st_size <= 62 << 20, 'bounded status input')
    codes = json.loads((Path(__file__).parent / 'result_codes.json').read_text(), object_pairs_hook=unique)
    return reconcile(p.read_text(encoding='utf-8',errors='strict'), pid, query, catalog_group, catalog, deadline, codes)
