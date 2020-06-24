from collections import OrderedDict

import torch


def remove_dup(lst, dct=None):

    nodupes = OrderedDict.fromkeys(lst)
    if dct.unk_index in nodupes:
        del nodupes[dct.unk_index]
    if nodupes:
        return list(nodupes.keys())
    else:
        return [dct.unk_index]


@torch.jit.script
def lengths_to_mask(lengths: torch.Tensor):
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    bsz = lengths.size(0)
    max_len = lengths.max()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
        .expand(bsz, max_len) < lengths.unsqueeze(1)
    return mask


@torch.jit.script
def index_lookup(indices, lookup):
    bat_dim = lookup.size(0)
    idx_dim = lookup.size(1)
    lookup = lookup.contiguous().view(bat_dim * idx_dim, lookup.size(2))
    shift = torch.arange(0, bat_dim * idx_dim, idx_dim, device=lookup.device, dtype=torch.int64).unsqueeze(1)
    return lookup[indices + shift]


@torch.jit.script
def reverse_padded_sequence(inputs: torch.Tensor, lengths: torch.Tensor, batch_first: bool = True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(1) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')

    tsz, bsz, _ = inputs.size()

    unreversed_indices = torch.arange(tsz, device=inputs.device, dtype=torch.int64)\
        .unsqueeze(1)\
        .expand(tsz, bsz)
    mask = (unreversed_indices < lengths.unsqueeze(0)).long()
    reversed_indices = - unreversed_indices + lengths.unsqueeze(0) - 1

    indices = reversed_indices * mask + unreversed_indices * (1 - mask)
    reversed_inputs = index_lookup(indices.transpose(0, 1), inputs.transpose(0, 1)).transpose(0, 1)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs