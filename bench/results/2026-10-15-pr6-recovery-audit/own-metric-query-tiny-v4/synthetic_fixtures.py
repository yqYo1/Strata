"""Offline synthetic data only; not a device catalog or approved evidence.
Repository LICENSE. Main's tests are the sole consumer. No I/O on import.
"""
import json
from status_reader import APIS, PREFIX

def fixture():
    rows = []; call_id = 0; pid = 12345
    def add(phase, **fields):
        rows.append(dict(schema=2,pid=pid,seq=len(rows)+1,phase=phase,**fields))
    def api(site, pre=None, post=None, code=0):
        nonlocal call_id
        call_id += 1
        add('entry',call=call_id,site=site,api=APIS[site],data=dict(present=True,**(pre or {})))
        add('result',call=call_id,site=site,api=APIS[site],code=code,hex=hex(code),
            name='ZE_RESULT_SUCCESS' if code==0 else 'ZE_RESULT_NOT_READY',
            data=dict(present=True,**(post or {})) if code==0 else None)
    catalog = [dict(metric=i,name='Synthetic'+str(i),units='events',result_type=0,
                    result_type_name='ZET_VALUE_TYPE_UINT32') for i in range(83)]
    g = dict(target=True,name='ComputeBasic',sampling_flags=1,domain=3,group=3,metric_count=83)
    add('open',route='Linux_EVENT_BASED_ComputeBasic')
    api('context-create',post=dict(context=100))
    api('group-count',dict(device=10,count=0),dict(count=24))
    api('group-list',dict(device=10,count=24),dict(count=24))
    for i in range(4):
        api('group-properties',dict(group=20+i,index=i),dict(count=83 if i==3 else 69,flags=1 if i==3 else 2,
            domain=3,group_name='ComputeBasic' if i in (2,3) else 'SyntheticOther'))
    api('activate',dict(context=100,device=10,group=23,count=1))
    api('pool-create',dict(context=500,device=10,group=23,count=64,type=0),dict(pool=200))
    for i in range(64): api('query-create',dict(pool=200,index=i),dict(query=1000+i))
    for instance in (11,12):
        api('query-begin',dict(list=900,query=1063))
        api('query-end',dict(list=900,query=1063,event=800+instance))
        add('submit',query=1063,instance=instance)
        pre = dict(query=1063,instance=instance,event=800+instance)
        api('query-ready',pre,code=1)
        api('query-ready',pre)
        api('data-size',dict(query=1063,instance=instance,size=0),dict(size=256))
        api('data-read',dict(query=1063,instance=instance,size=256),dict(size=256))
        api('query-reset',dict(query=1063))
        add('collected',query=1063,instance=instance)
    api('schema-group',dict(group=23),dict(count=83,flags=1,domain=3,group_name='ComputeBasic'))
    api('schema-list',dict(group=23,count=83),dict(count=83))
    for i,m in enumerate(catalog):
        api('schema-metric',dict(metric=3000+i),dict(metric_name=m['name'],units=m['units'],result_type=m['result_type']))
    for instance in (11,12):
        pre=dict(group=23,instance=instance,size=256,samples=0,values=0)
        api('calculate-size',pre,dict(samples=1,values=83))
        api('calculate-read',{**pre,'samples':1,'values':83},dict(samples=1,values=83,sample0=83))
    api('deactivate',dict(context=100,device=10,count=0))
    api('context-destroy',dict(context=100))
    for i in range(64): api('query-destroy',dict(query=1000+i))
    api('pool-destroy',dict(pool=200))
    add('close',calls=call_id)
    query=dict(decoded=True,rows=[dict(global_instance_id=i) for i in (11,12)])
    return rows,pid,query,g,catalog

def encode(rows):
    return ''.join(PREFIX+json.dumps(row,separators=(',',':'))+'\n' for row in rows)

def resequence(rows):
    for i,row in enumerate(rows): row['seq']=i+1
    return rows
