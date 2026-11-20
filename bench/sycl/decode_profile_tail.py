"""Generate a global decode ranking with the measured 8192-token prefill scratch extent."""

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(PROJECT / 'tools'), str(PROJECT / 'tools/sycl')]
import cpu_work_census as census
import make_profile

require = census.guards.require
PACK = 'd9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d'
# Counts by ascending aligned slot size. Each vector retains as many original
# first-132 members as possible while matching the measured scratch tail.
PREFIX = {
    '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf': (33, 9, 0, 49, 36, 4, 1),
    'ddddf9b361dd1cb1e68ef8d130759fa15da3b84d74f21d37b28f377493b80bbc': (40, 9, 0, 34, 32, 17, 0),
}


def profile_for_loan(profile, manifest):
    key = hashlib.sha256(profile.read_bytes()).hexdigest()
    require(key in PREFIX and hashlib.sha256(manifest.read_bytes()).hexdigest() == PACK,
            'this slot plan requires the measured IQ3_S pack and one of its two shipped rankings')
    census.expected_membership(profile)
    ranked = make_profile.read_profile(profile)
    sizes = [(size + 255) // 256 * 256 for _, _, size in census.manifest_formats(manifest)]
    reference = [size for layer, size in enumerate(sizes) for _ in range(34 if layer == 47 else 26)]
    first = 132
    target = sum(reference[first:])
    cap = sum(reference) // max(sizes) * max(sizes)

    used = 0
    admitted = []
    for pair in ranked:
        size = sizes[pair[0]]
        if used + size > cap:
            break
        used += size
        admitted.append(pair)
    remaining = dict(zip(sorted(set(sizes)), PREFIX[key], strict=True))
    head, tail = [], []
    for pair in admitted:
        size = sizes[pair[0]]
        if remaining[size]:
            remaining[size] -= 1
            head.append(pair)
        else:
            tail.append(pair)
    require(len(head) == first and not any(remaining.values()), 'prefix size classes do not fit')
    # The first slot in the loan must also match. Dropping it gives the
    # original too-small tail, so the allocator chooses exactly first=132.
    pivot = next(i for i, pair in enumerate(tail) if sizes[pair[0]] == reference[first])
    tail.insert(0, tail.pop(pivot))
    require(sum(sizes[l] for l, _ in tail) == target and target == 2321664000
            and set(head + tail) == set(admitted)
            and used <= cap < used + sizes[ranked[len(admitted)][0]],
            'rank membership, global capacity or prefill workspace changed')
    return head + tail + ranked[len(admitted):], len(admitted), target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    ranked, slots, tail_bytes = profile_for_loan(args.profile, args.manifest)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory) / 'expert-profile.bin'
        make_profile.write_profile(tmp, ranked)
        census.expected_membership(tmp)  # validates every pair and the inverse slot table
        require(make_profile.read_profile(tmp) == ranked, 'serialized order differs')
        payload = tmp.read_bytes()
    if args.out:
        with args.out.open('xb') as out:
            out.write(payload)
    print(f'global slots={slots} first=132 tail_bytes={tail_bytes} sha256={hashlib.sha256(payload).hexdigest()}')


if __name__ == '__main__':
    main()
