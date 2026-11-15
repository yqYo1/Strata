#!/usr/bin/env python3
"""Fresh bank cohort after the original host-fixture failure; same native unit."""
import fcntl
import json
import sys

import prefill_bank_compare as campaign

# Separate controller process/context; never mutate native process environment.
# Keep v1source unchanged while its independent read-only source audit runs.
campaign.ROOT=campaign.B/'prefill-bank512-compare-v2'
campaign.KEY='prefill_bank512_compare_v2'
campaign.THIS='tools/sycl/prefill_bank_compare_v2.py'

if __name__=='__main__':
    campaign.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','host','0','1','2','3','4','5'),
        'prepare|host|0..5')
    with (campaign.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign.prior.admission.pti.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare':campaign.prepare()
        else:
            with (campaign.ROOT/(action+'-attempt.json')).open('x') as stream:
                json.dump(dict(action=action),stream)
            if action=='host':campaign.host()
            else:campaign.prior.runner.run(int(action),campaign)
