"""Source authoring only: deterministic unified diff, no patch apply/build/test.
Repository LICENSE. Main may regenerate after reviewing the pinned source inputs.
Writes only this directory; original PTI is read-only.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
D = Path(__file__).resolve().parent
PTI = Path('/home/yayoi/ghq/github.com/intel/pti-gpu')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-unitrace-build-root-v1/build')
relative = 'tools/unitrace/src/levelzero/ze_collector.h'
source = (PTI / relative).read_text()
if hashlib.sha256(source.encode()).hexdigest() != '6ee0130e2b03918a5e4bb5d310dacb94e5ba06bf67c21a23d3134dc406aa9122':
    raise RuntimeError('immutable baseline differs')
header = (B / 'level_zero/ze_api.h').read_text()
if hashlib.sha256(header.encode()).hexdigest() != '095a0eb442ae21946f29147c209f270b93dd0eb5b12a2d0c67e5f4138b533712':
    raise RuntimeError('immutable enum header differs')
result_enum = header.split('typedef enum _ze_result_t\n{', 1)[1].split('} ze_result_t;', 1)[0]
enums = re.findall(r'^\s*(ZE_RESULT_[A-Z0-9_]+)\s*=\s*(0x[0-9a-f]+|[0-9]+)', result_enum, re.M)
# Official API identifiers/numeric values; no copied comments/header implementation.
(D / 'metric_query_result_names.inc').write_text('// Official Level Zero enum identifiers; repository LICENSE. See provenance.\n' +
    ''.join('OWN_RESULT(' + name + ')\n' for name, _ in enums))
(D / 'result_codes.json').write_text(json.dumps({name:int(value, 0) for name,value in enums}, indent=2) + '\n')

def f(name, value): return 'OwnMetricAudit::field("' + name + '", ' + value + ')'
def h(name, value): return f(name, 'OwnMetricAudit::id(' + value + ')')
def t(name, value): return 'OwnMetricAudit::array_text("' + name + '", ' + value + ')'
def fields(*args): return ' + '.join(args) if args else 'std::string()'
# Exact original call-site line -> site, entry fields, success-result fields.
# Outputs of failed APIs are deliberately not read.
sites = {}
def put(line, site, pre, post): sites[line] = (site, pre, post)
put(95,'query-destroy',h('query','it->first'), '')
put(103,'pool-destroy',h('pool','*it'), '')
for line in (128,161):
    put(line,'pool-create',fields(h('context','context'),h('device','device'),h('group','group'),f('count','desc.count'),f('type','desc.type')),h('pool','pool'))
for line,index in ((137,'i'),(145,'pool_size_ - 1'),(169,'i'),(177,'pool_size_ - 1')):
    put(line,'query-create',fields(h('pool','pool'),f('index',index)),h('query','query'))
put(215,'query-reset',h('query','query'),'')
put(1443,'deactivate',fields(h('context','it->first'),h('device','it->second'),f('count','0')),'')
put(1454,'context-destroy',h('context','context'),'')
# Context creation occurs only in the existing options_.metric_query branch.
put(1940,'context-create','',h('context','context'))
put(2004,'group-count',fields(h('device','device'),f('count','num_groups')),f('count','num_groups'))
put(2011,'group-list',fields(h('device','device'),f('count','num_groups')),f('count','num_groups'))
props = fields(f('count','group_props.metricCount'),f('flags','group_props.samplingType'),f('domain','group_props.domain'),t('group_name','group_props.name'))
put(2020,'group-properties',fields(h('group','groups[k]'),f('index','k')),props)
put(2042,'activate',fields(h('context','context'),h('device','device'),h('group','group'),f('count','1')),'')
put(2236,'schema-group',h('group','group'),props)
put(2252,'schema-list',fields(h('group','group'),f('count','metric_count')),f('count','metric_count'))
put(2260,'schema-metric',h('metric','metric'),fields(t('metric_name','metric_props.name'),t('units','metric_props.resultUnits'),f('result_type','metric_props.resultType')))
q = 'command_metric_query->metric_query_'
common = fields(h('query',q),f('instance','command_metric_query->instance_id_'))
put(2277,'query-ready',fields(common,h('event','command_metric_query->metric_query_event_')),'')
put(2283,'data-size',fields(common,f('size','size')),f('size','size'))
put(2289,'data-read',fields(common,f('size','size2')),f('size','size2'))
for line,site in ((2592,'calculate-size'),(2601,'calculate-read')):
    pre=fields(h('group','group'),f('instance','it->second.instance_id_'),f('size','it->second.metrics_->size()'),f('samples','num_samples'),f('values','num_metrics'))
    post=fields(f('samples','num_samples'),f('values','num_metrics'))
    if site=='calculate-read': post=fields(post,f('sample0','samples.empty() ? 0 : samples[0]'))
    put(line,site,pre,post)
put(3596,'query-begin',fields(h('list','command_list'),h('query','ze_instance_data.query_')),'')
for line in (3706,3856,3991,4128,4253):
    put(line,'query-end',fields(h('list','command_list'),h('query','query'),h('event','metric_query_event')),'')

pattern = re.compile(r'ZE_FUNC\((\w+)\)\(')
edits=[]; proof=[]
for match in pattern.finditer(source):
    line=source.count('\n',0,match.start())+1
    if line not in sites: continue
    site,pre,post=sites.pop(line)
    api=match[1]; pos=match.end(); depth=1
    # Arguments at the selected exact sites contain no string/character literals.
    while depth:
        if source[pos]=='(': depth+=1
        elif source[pos]==')': depth-=1
        pos+=1
    args=source[match.end():pos-1]
    replacement='OwnMetricAudit::call("'+site+'", "'+api+'", ZE_FUNC('+api+'), '+(pre or 'std::string()')+', [&] { return '+(post or 'std::string()')+'; }, '+args+')'
    edits.append((match.start(),pos,replacement))
    proof.append(dict(original_line=line,api=api,site=site,original_call=source[match.start():pos]))
if sites: raise RuntimeError('authoring call-site mismatch: '+repr(sites))
modified=source
for start,end,replacement in reversed(edits): modified=modified[:start]+replacement+modified[end:]
def replace(old,new,count=1):
    global modified
    if modified.count(old)!=count: raise RuntimeError('authoring anchor mismatch: '+old[:100])
    modified=modified.replace(old,new)
replace('#include "logger_factory.h"', '#include "logger_factory.h"\n#include "metric_query_audit.h"')
replace('    data_dir_name_ = data_dir_name;', '''    OwnMetricAudit::enable(options_.metric_query && !options_.metric_stream && !options_.stall_sampling &&
        utils::GetEnv("UNITRACE_MetricGroup") == "ComputeBasic");
    data_dir_name_ = data_dir_name;''')
replace('      status = OwnMetricAudit::call("query-destroy"', '      OwnMetricAudit::safe_query(OwnMetricAudit::id(it->first));\n      status = OwnMetricAudit::call("query-destroy"')
replace('    free_pool_.clear();\n#endif /* _WIN32 */', '    free_pool_.clear();\n    OwnMetricAudit::close();\n#endif /* _WIN32 */')
replace('    const std::lock_guard<std::mutex> lock(query_pool_mutex_);\n    auto it = query_pool_map_.find(query);', '    OwnMetricAudit::safe_query(OwnMetricAudit::id(query));\n    const std::lock_guard<std::mutex> lock(query_pool_mutex_);\n    auto it = query_pool_map_.find(query);')
replace('      metric_queries_submitted_.push_back(query);', '      OwnMetricAudit::submitted(OwnMetricAudit::id(query->metric_query_), query->instance_id_);\n      metric_queries_submitted_.push_back(query);')
replace('    if (options_.metric_query) {\n      for (auto it = metric_activations_', '    if (options_.metric_query) {\n      OwnMetricAudit::safe_finalize();\n      for (auto it = metric_activations_')
replace('              uint32_t num_groups = 0;', '              uint32_t num_groups = 0;')
replace('              if (num_groups > 0) {', '              OwnMetricAudit::require(num_groups > 0 && num_groups <= 256, "group_count_outside_tiny_scope");\n              if (num_groups > 0) {')
replace('                std::vector<zet_metric_group_handle_t> groups(num_groups, nullptr);', '                const uint32_t original_groups = num_groups;\n                std::vector<zet_metric_group_handle_t> groups(num_groups, nullptr);')
replace('                for (uint32_t k = 0; k < num_groups; ++k) {', '                OwnMetricAudit::require(num_groups == original_groups, "changed_group_list_size");\n                for (uint32_t k = 0; k < num_groups; ++k) {')
replace('                if (group == nullptr) {', '                OwnMetricAudit::require(group != nullptr, "requested_event_group_unavailable");\n                if (group == nullptr) {')
replace('                    group = groups[k];', '                    OwnMetricAudit::require(group_props.samplingType == 1 && group_props.metricCount == 83, "selected_group_schema_outside_tiny_scope");\n                    group = groups[k];')
replace('    return group_props.metricCount;', '    OwnMetricAudit::require(group_props.metricCount == 83, "schema_count_outside_tiny_scope");\n    return group_props.metricCount;')
replace('    PTI_ASSERT(metric_count == metrics.size());', '    OwnMetricAudit::require(metric_count == metrics.size(), "changed_schema_list_size");\n    PTI_ASSERT(metric_count == metrics.size());')
replace('        if (collect_data) {\n          size_t size = 0;', '        OwnMetricAudit::require(collect_data, "query_collection_disabled");\n        if (collect_data) {\n          size_t size = 0;')
replace('          if ((status == ZE_RESULT_SUCCESS) && (size > 0)) {', '          OwnMetricAudit::require(size > 0 && size <= (62u << 20), "empty_or_oversized_query_data");\n          if ((status == ZE_RESULT_SUCCESS) && (size > 0)) {')
replace('            if (size2 == size) {', '            OwnMetricAudit::require(size2 == size, "changed_query_data_size");\n            if (size2 == size) {')
replace('      query_pools_.ResetQuery(command_metric_query->metric_query_);', '      query_pools_.ResetQuery(command_metric_query->metric_query_);\n      OwnMetricAudit::completed(OwnMetricAudit::id(command_metric_query->metric_query_), command_metric_query->instance_id_);')
replace('        if ((status == ZE_RESULT_SUCCESS) && (num_samples > 0) && (num_metrics > 0)) {', '        OwnMetricAudit::require(num_samples == 1 && num_metrics == 83, "calculation_size_outside_tiny_scope");\n        if ((status == ZE_RESULT_SUCCESS) && (num_samples > 0) && (num_metrics > 0)) {')
replace('          if (status == ZE_RESULT_SUCCESS) {\n            std::string str;', '          OwnMetricAudit::require(num_samples == 1 && num_metrics == 83 && samples[0] == 83, "changed_calculation_shape");\n          if (status == ZE_RESULT_SUCCESS) {\n            std::string str;')
# This anchor has only the selected query begin call preceding it.
replace('      status = OwnMetricAudit::call("query-begin",', '      OwnMetricAudit::begin(OwnMetricAudit::id(ze_instance_data.query_));\n      status = OwnMetricAudit::call("query-begin",')
# Original begin failures now _Exit in call; busy state is intentionally retained.
patch=''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative,n=3))
for name in ('metric_query_audit.h','metric_query_result_names.inc'):
    patch+=''.join(difflib.unified_diff([], (D/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/tools/unitrace/src/levelzero/'+name,n=3))
(D/'pti-query-status.patch').write_text(patch)
(D/'call-sites.json').write_text(json.dumps(proof,indent=2)+'\n')
(D/'patched-source-sha256.json').write_text(json.dumps({relative:hashlib.sha256(modified.encode()).hexdigest(),**{
    'tools/unitrace/src/levelzero/'+n:hashlib.sha256((D/n).read_bytes()).hexdigest()
    for n in ('metric_query_audit.h','metric_query_result_names.inc')}},indent=2)+'\n')
