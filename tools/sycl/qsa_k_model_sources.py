"""Own MIT production overlay for the qualified direct-K component.

The model comparison starts from exact selected6e40 source, not the current
tree's unselected native changes. Only the ordered,non-probe prefill path changes.
"""
import hashlib
import textwrap

import qsa_k_loop_sources as component

BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'


def production_prefix(original):
    if hashlib.sha256(original).hexdigest() != component.PREFIX_SHA256:
        raise ValueError('require exact selected production prefix')
    source = original.decode()
    branch = ('          if constexpr (OrderedExp && !CaptureState) {\n' +
              textwrap.indent(component.REPLACEMENT, '  ') + '\n' +
              '          } else {\n' + textwrap.indent(component.ORIGINAL, '  ') +
              '\n          }')
    result = component.once(source, component.ORIGINAL, branch)
    if component.once(result, branch, component.ORIGINAL) != source:
        raise ValueError('production overlay changed another source region')
    return result.encode()
