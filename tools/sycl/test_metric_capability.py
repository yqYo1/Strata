#!/usr/bin/env python3
"""Host rejection controls for the direct-C topology evidence parser."""
import copy
import io
import json
import sys
import unittest

import metric_capability as capability


class TopologyEvidence(unittest.TestCase):
    def setUp(self):
        self.device = dict(pci='0000:05:00.0', vendor_id=0x8086,
                           device_id=0xe20c, name='B570', uuid_hex='ab'*16)
        self.nodes = [dict(kind='topology_node', node=0, parent_node=None,
                           device_id=0xe20c, name='B570', uuid_hex='ab'*16)]
        self.groups = []
        self.complete = dict(scope='topology_only', root_nodes=1, child_nodes=0,
                             groups=0, event_groups=0, collection_qualified=False,
                             hardware_utilization_measured=False)

    def parse(self):
        return capability.topology_inventory(self.nodes+self.groups, self.device, self.complete)

    def add_child(self):
        self.nodes.append(dict(self.nodes[0], node=1, parent_node=0, uuid_hex='cd'*16))
        self.complete['child_nodes'] = 1
        self.groups.append(dict(kind='topology_group', node=1, group=0, name='combined',
                                sampling_flags=3, event_based=True, time_based=True,
                                metric_count=83))
        self.complete.update(groups=1, event_groups=1)

    def test_zero_children_and_groups_are_valid_negative_evidence(self):
        result = self.parse()
        self.assertEqual((result['child_nodes'], result['child_event_groups']), (0, 0))

    def test_combined_sampler_bits_and_child_scope(self):
        self.add_child()
        self.assertEqual(self.parse()['child_event_groups'], 1)

    def test_wrong_physical_root_is_refused(self):
        self.device['pci'] = '0000:06:00.0'
        with self.assertRaises(RuntimeError):
            self.parse()

    def test_missing_child_record_is_refused(self):
        self.complete['child_nodes'] = 1
        with self.assertRaises(RuntimeError):
            self.parse()

    def test_duplicate_node_is_refused(self):
        self.nodes.append(copy.deepcopy(self.nodes[0]))
        self.complete['child_nodes'] = 1
        with self.assertRaises(RuntimeError):
            self.parse()

    def test_group_cannot_belong_to_absent_child(self):
        self.add_child()
        self.groups[0]['node'] = 2
        with self.assertRaises(RuntimeError):
            self.parse()

    def test_sampling_bool_must_match_bitmask(self):
        self.add_child()
        self.groups[0]['sampling_flags'] = 2
        with self.assertRaises(RuntimeError):
            self.parse()

    def test_group_count_must_conserve(self):
        self.add_child()
        self.complete['groups'] = 0
        with self.assertRaises(RuntimeError):
            self.parse()

    def test_collection_claim_is_refused(self):
        self.complete['collection_qualified'] = True
        with self.assertRaises(RuntimeError):
            self.parse()


if __name__ == '__main__':
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TopologyEvidence))
    print(json.dumps(dict(cases=result.testsRun, failures=len(result.failures),
                          errors=len(result.errors), passed=result.wasSuccessful())))
    if not result.wasSuccessful():
        sys.stderr.write(transcript.getvalue())
        sys.exit(1)
