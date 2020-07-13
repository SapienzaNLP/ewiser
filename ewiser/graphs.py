import itertools

import networkx as nx

from ewiser.fairseq_ext.data.dictionaries import ResourceManager, fix_offset

def read_graph(*paths, input_keys=None):

    self_loops_count = 0
    offsets = ResourceManager.get_offsets_dictionary()

    if input_keys is None:
        with open(paths[0]) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                else:
                    offset1, offset2, *info = line.split()
                    if offset1.startswith('bn:'):
                        input_keys = 'bnids'
                    elif offset1.startswith('wn:'):
                        input_keys = 'offsets'
                    else:
                        input_keys = 'sensekeys'
                    break
    assert input_keys is not None

    remap = None
    g = nx.DiGraph()

    for path in paths:

        with open(path) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                offset1, offset2, *info = line.split()
                if 0 <= len(info) <= 1:
                    w = None
                else:
                    try:
                        w = float(info[1])
                    except ValueError:
                        w = None

                if offset1.startswith('bn:'):
                    if remap is None:
                        remap = ResourceManager.get_bnids_to_offset_map()
                    offsets1 = remap.get(offset1)
                elif offset1.startswith('wn:'):
                    offsets1 = [offset1]
                else:
                    raise NotImplementedError

                if offset2.startswith('bn:'):
                    if remap is None:
                        remap = ResourceManager.get_bnids_to_offset_map()
                    offsets2 = remap.get(offset2)
                elif offset2.startswith('wn:'):
                    offsets2 = [offset2]
                else:
                    raise NotImplementedError

                for offset1, offset2 in itertools.product(offsets1, offsets2):
                    offset1 = fix_offset(offset1) # v -> child in hypernymy
                    offset2 = fix_offset(offset2) # u -> father in hypernymy
                    trg_node = offsets.index(offset1)
                    src_node = offsets.index(offset2)
                    g.add_edge(src_node, trg_node, w=w)
                    self_loops_count += int(src_node == trg_node)

    return g
