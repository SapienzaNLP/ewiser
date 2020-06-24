import math

import torch
from torch.nn import functional as F
from torch.nn import _reduction

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import utils

from ewiser.fairseq_ext.data.dictionaries import TargetManager

def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def smoothed_nll_loss(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                      smooth_eps=None, smooth_dist=None, size_average=None, reduce=None):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""

    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)

    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction, size_average=size_average, reduce=reduce)

    lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


@register_criterion('weighted_cross_entropy')
class WeightedCrossEntropyCriterionWithSoftmaxMask(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        if hasattr(task, "criterion_weights") and task.criterion_weights is not None:
            self.weight = torch.nn.Parameter(task.criterion_weights)
            self.weight.requires_grad = False
        else:
            self.weight = None

        self.nspecial = task.target_dictionary.nspecial

        self.only_use_targets = getattr(args, 'only_use_targets', False)
        self.label_smoothing = getattr(args, 'label_smoothing', 0.)
        assert 0. <= self.label_smoothing <= 1.
        self.trg_manager = TargetManager(args.kind)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        ntokens_orig = sample['ntokens']

        use_only_targets = self.only_use_targets and not sample.get('dummy', False)
        if use_only_targets:
            sample = self.trg_manager.remove_non_targets(sample, self.nspecial)
        
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        results, _ = self.trg_manager.calulate_metrics(lprobs, sample)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = self.trg_manager.get_targets(sample).view(-1)

        if self.label_smoothing:
            loss = smoothed_nll_loss(
                lprobs,
                target,
                weight=self.weight,
                size_average=False,
                ignore_index=self.padding_idx,
                reduce=reduce,
                smooth_eps=self.label_smoothing,
            )
        else:
            loss = F.nll_loss(
                lprobs,
                target,
                weight=self.weight,
                size_average=False,
                ignore_index=self.padding_idx,
                reduce=reduce
            )

        if use_only_targets:
            loss *= sample['ntokens'] / ntokens_orig

        sample_size = self.trg_manager.get_targets(sample).size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'hit': results['hit'],
            'tot': results['tot'],
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': self.trg_manager.get_targets(sample).size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        try:
            acc = sum(log.get('hit') for log in logging_outputs) / sum(log.get('tot') for log in logging_outputs)
        except ZeroDivisionError:
            acc = 0.
        agg_output = {
            'acc': acc,
            #'hit': sum(log.get('hit') for log in logging_outputs),
            #'tot': sum(log.get('tot') for log in logging_outputs),
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
