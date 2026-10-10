#!/usr/bin/env python3
"""Host-only contract harness including the real diagnostic ledger header."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CPP = r'''
#include "strata/prefill_service_ledger.hpp"
#include <stdexcept>
#include <string>
static int ops = 0;
struct Event { uint64_t t[3] = {10,20,30}; bool complete = true; };
struct Queue { bool property = true, aspect = true, fault = false; };
static int query_fail = -1;
static bool unknown = false;
struct Ops {
 static bool profiling(Queue& q) { ++ops; return q.property && q.aspect; }
 static bool complete(const Event& e) { ++ops; if (unknown) throw std::runtime_error("unknown"); return e.complete; }
 static void async_errors(Queue& q) { ++ops; if(q.fault) throw std::runtime_error("async"); }
 static uint64_t timestamp(const Event& e,int i) { ++ops; if(i==query_fail) throw std::runtime_error("profile"); return e.t[i]; }
};
struct Canary { ~Canary() { std::puts("UNWOUND"); } };
using Ledger = strata::prefill::diagnostic::ServiceLedger<Event,Queue,Ops>;
using Id = strata::prefill::diagnostic::ServiceId;
int main(int argc, char** argv) {
 const std::string mode = argv[1];
 Canary canary;
 Queue q, other;
 if(mode=="property") q.property=false;
 if(mode=="aspect") q.aspect=false;
 {
 Ledger ledger(mode!="off","test");
 if(mode=="cap") ledger.event_cap=1;
 if(mode=="buckets") ledger.bucket_cap=1;
 ledger.admit(&q);
 Id id; id.chunk=0; id.layer=1; id.expert=3; id.rows=7; id.n=11; id.k=13; id.ldy=17;
 Event a,b;
 if(mode=="full_chunk") {
  for(int layer=0;layer<48;++layer) for(int expert=0;expert<512;++expert) {
   id.layer=layer; id.expert=expert; id.rows=160;
   id.role=0; id.n=1280; id.k=2560; id.ldy=1280;
   ledger.expect_gemm(id); ledger.attempt(); ledger.record(a,&q,id,0,0);
   id.role=1; id.n=2560; id.k=640; id.ldy=2560;
   ledger.attempt(); ledger.record(b,&q,id,0,0);
  }
  ledger.fold(); ledger.report(); return 0;
 }
 if(mode=="backward") a.t[2]=1;
 if(mode=="equal") a.t[0]=a.t[1]=a.t[2]=10;
 if(mode=="pipeline") { a.t[2]=100; b.t[0]=11; b.t[1]=12; b.t[2]=13; }
 if(mode=="incomplete" || mode=="destructor") a.complete=false;
 if(mode=="unknown") unknown=true;
 if(mode=="async") q.fault=true;
 if(mode=="query") query_fail=1;
 if(mode=="invalid_id") id.n=-1; // Expectation is valid; reject the submitted shape.
 if(mode=="expected_invalid") id.layer=-1;
 if(mode=="unsupported") ledger.invalidate("unsupported_path");
 if(mode=="missing") ledger.attempt();
 if(mode=="expected_missing") ledger.expect_gemm(id);
 if(mode!="zero" && mode!="missing" && mode!="expected_missing" && mode!="bytes") {
  ledger.expect_gemm(id);
  ledger.attempt(); ledger.record(a,&q,id,0,0.25,0.5,0.75);
  if(mode=="destructor") return 0;
  if(mode=="partial") { ledger.fold(); ledger.report(); return 0; }
  id.role=1; id.n=13; id.k=11; id.ldy=13;
  if(mode=="rows") id.rows=8;
  if(mode=="duplicate") id.role=0;
  ledger.attempt(); ledger.record(b,mode=="queue"?&other:&q,id,0,0.25);
 }
 if(mode=="bytes") {
  // Copy-only records exercise unsigned accumulation rather than GEMM pairs.
  id.role=2; id.source=1;
  ledger.attempt(); ledger.record(a,&q,id,UINT64_MAX,0.0);
  ledger.attempt(); ledger.record(b,&q,id,1,0.0);
 }
 ledger.fold();
 if(mode=="normal") {
  id.role=0; id.rows=7; id.n=11; id.k=13; id.ldy=17; id.chunk=64;
  ledger.expect_gemm(id);
  ledger.attempt(); ledger.record(a,&q,id,0,0.1);
  id.role=1; id.n=13; id.k=11; id.ldy=13;
  ledger.attempt(); ledger.record(b,&q,id,0,0.1);
  ledger.fold();
 }
 ledger.report();
 }
 std::printf("OPS %d\n",ops);
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix='strata-service-ledger-') as tmp:
        source, exe = Path(tmp)/'contract.cpp', Path(tmp)/'contract'
        source.write_text(CPP)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-O0',
                        '-I', str(ROOT/'include'), str(source), '-o', str(exe)], check=True)
        invalid = {'property':'profiling_unavailable', 'aspect':'profiling_unavailable',
                   'backward':'backward_event_timestamps', 'query':'profiling_query_failure',
                   'queue':'queue_changed', 'cap':'event_cap_overflow', 'buckets':'aggregate_cap_overflow',
                   'rows':'gemm_pair_identity', 'duplicate':'gemm_pair_identity',
                   'invalid_id':'record_identity_or_host_time', 'expected_invalid':'expected_identity', 'bytes':'arithmetic_overflow',
                   'unsupported':'unsupported_path', 'missing':'final_reconciliation',
                   'expected_missing':'routed_pair_reconciliation', 'partial':'gemm_pair_incomplete'}
        fatal = {'incomplete':'event_incomplete','unknown':'completion_unknown',
                 'async':'async_queue_failure','destructor':'destruction_event_incomplete'}
        for mode in list(invalid)+list(fatal)+['normal','equal','pipeline','off','zero','full_chunk']:
            result = subprocess.run([str(exe),mode],capture_output=True,text=True)
            receipts = [json.loads(x.split(': ',1)[1]) for x in result.stderr.splitlines()
                        if x.startswith('strata prefill service validity: ')]
            aggregates = [json.loads(x.split(': ',1)[1]) for x in result.stderr.splitlines()
                          if x.startswith('strata prefill returned-event ledger: ')]
            if mode=='off':
                assert result.returncode==0 and not receipts and not aggregates and 'OPS 0' in result.stdout
                continue
            assert len(receipts)==1, (mode,result)
            receipt=receipts[0]
            if mode in invalid or mode in fatal:
                assert receipt['status']=='invalid' and not aggregates, (mode,result)
                assert receipt['reason']=={**invalid,**fatal}[mode], (mode,receipt)
                assert result.returncode==(1 if mode in fatal else 0), (mode,result)
                if mode in fatal:
                    assert 'UNWOUND' not in result.stdout and 'OPS' not in result.stdout, (mode,result)
                if mode=='query':
                    assert receipt['query_attempts']==6 and receipt['query_failures']==2
                    assert 'field 1: profile' in result.stderr
                if mode=='backward':
                    assert 'submit 10 start 20 end 1' in result.stderr
                    assert receipt['backward']==1
                    assert receipt['rejected_submit_ns']==10 and receipt['rejected_start_ns']==20 and receipt['rejected_end_ns']==1
                if mode=='cap': assert receipt['retained']==1 and receipt['dropped']==1
                continue
            assert result.returncode==0 and receipt['status']=='valid' and len(aggregates)==1, (mode,result)
            count=49152 if mode=='full_chunk' else 4 if mode=='normal' else 0 if mode=='zero' else 2
            assert receipt['attempts']==receipt['submitted']==receipt['retained']==receipt['folded']==count
            assert receipt['query_attempts']==receipt['query_successes']==3*count
            assert receipt['dropped']==receipt['pending']==receipt['query_failures']==0
            assert aggregates[0]['calls']==count
            assert receipt['expected_registered']==receipt['completed_role_pairs']==count//2
            assert receipt['expected_pending']==0
            assert sum(x['calls'] for x in aggregates[0]['coverage'])==count
            assert sum(x['calls'] for x in aggregates[0]['shapes'])==count
            if mode=='pipeline': assert aggregates[0]['returned_event_ms']==81/1e6
        print('Returned-event ledger CPU contracts passed.')


if __name__=='__main__':
    main()
