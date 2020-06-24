import torch
import os

from pathlib import Path

# TODO: remove this variable
DEHARDCODE_ALPHA = float(os.environ.get('LOGIT_CONV_ALPHA', 1.0))

def unpack_sparse_tensor(sparse_tensor):
    pieces = (sparse_tensor._indices(), sparse_tensor._values(), torch.LongTensor(list(sparse_tensor.shape)))
    return pieces

def repack_sparse_tensor(*pieces):
    ii, vv, size = pieces
    if isinstance(size, torch.Size):
        ...
    else:
        size = torch.Size(size.cpu().tolist())
    return torch.sparse.FloatTensor(ii, vv, size).coalesce()

class StructuredLogits(torch.nn.Module):

    def __init__(self, adjacency=None, adjacency_trainable=False, renormalize=False):
        super().__init__()

        import torch_scatter
        import torch_sparse

        self._torch_scatter = torch_scatter
        self._torch_sparse = torch_sparse

        self.adjacency = None
        self.adjacency_pars = None
        self.renormalize = renormalize

        if adjacency is not None:

            ii, vv, size = unpack_sparse_tensor(adjacency)
            ii = torch.nn.Parameter(ii, requires_grad=False)
            vv = torch.nn.Parameter(vv, requires_grad=adjacency_trainable)
            size = torch.nn.Parameter(size, requires_grad=False)

            self.adjacency_pars = torch.nn.ParameterList([ii, vv, size])
            self._coalesce(self.adjacency_pars)

            if not self.renormalize:
                self._initialize_to_1_over_n(self.adjacency_pars)

            self.self_loops = None

        else:
            self.adjacency_pars = None

    def forward(self, logits):

        if self.adjacency_pars is not None:

            if DEHARDCODE_ALPHA > 0:

                logits_old = logits
                neighbors = self._spmm(logits, self.adjacency_pars)

                if self.renormalize:
                    neighbor_sum = self._get_row_sum(self.adjacency_pars)
                    neighbors = neighbors / neighbor_sum.view(1, 1, -1)

                logits = neighbors + logits_old

        return logits

    @classmethod
    def _read_and_load_in_args(cls, args):

        args.decoder_structured_logits_edgelists = getattr(args, 'decoder_structured_logits_edgelists', [])
        adjacency = None

        if getattr(args, 'decoder_use_structured_logits', False):

            assert args.decoder_structured_logits_edgelists, 'No edges provided!'

            if args.decoder_structured_logits_edgelists:
                if isinstance(args.decoder_structured_logits_edgelists[0], torch.Tensor):
                    adjacency = repack_sparse_tensor(*args.decoder_structured_logits_edgelists)
                else:
                    from ewiser.fairseq_ext.data.dictionaries import ResourceManager
                    adjacency = ResourceManager.make_adjacency_from_files(*args.decoder_structured_logits_edgelists)
                    args.decoder_structured_logits_edgelists = unpack_sparse_tensor(adjacency.clone().cpu())
            else:
                adjacency = None

        return {
            'adjacency': adjacency,
        }

    def _coalesce(self, params):
        ii, vv, size = params
        coalesced_ii, coalesced_vv = self._torch_sparse.coalesce(ii, vv, *size, op='max')
        ii.data = coalesced_ii
        vv.data = coalesced_vv
        return params

    def _get_row_sum(self, params):
        ii, vv, size = params
        row_sum = self._torch_scatter.scatter_add(vv, ii[0], dim_size=size[1])
        return row_sum

    def _get_col_sum(self, params):
        ii, vv, size = params
        col_sum = self._torch_scatter.scatter_add(vv, ii[1], dim_size=size[0])
        return col_sum

    def _initialize_to_1_over_n(self, params, sum='none'):
        if sum == 'none':
            return params
        ii, vv, size = params
        vv.data[:] = 1
        if sum == 'row':
            row_sum = self._get_row_sum(params)
            vv.data[:] = 1 / row_sum[ii[0]]
        elif sum == 'col':
            col_sum = self._get_col_sum(params)
            vv.data[:] = 1 / col_sum[ii[1]]
        return params

    def _spmm(self, inp, params):
        ii, vv, size = params
        old_inp_size = inp.size()
        inp_flat_T = inp.view(-1, inp.size(-1)).t()  # H x D_0*D_1*...*D_n
        out_flat = self._torch_sparse.spmm(
            ii, vv,
            m=size[0], n=size[1],
            matrix=inp_flat_T
        ).t()
        out = out_flat.view(*old_inp_size)
        return out

    @property
    def device(self):
        return next(self.parameters()).device
