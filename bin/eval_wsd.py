import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from argparse import ArgumentParser

from fairseq.data import Dictionary
from torch.utils.data import DataLoader

from ewiser.fairseq_ext.data.dictionaries import ResourceManager, SequenceLabelingTaskKind, TargetManager, MFSManager, \
    DEFAULT_DICTIONARY
from ewiser.fairseq_ext.data.utils import patched_lemma_from_key, make_offset
from ewiser.fairseq_ext.data.wsd_dataset import WSDDataset, WSDConcatDataset
from ewiser.fairseq_ext.models.sequence_tagging import LinearTaggerEnsembleModel
from ewiser.fairseq_ext.tasks.sequence_tagging import SequenceLabelingTask


def add_bias_if_missing(checkpoint, path=''):

    weight: torch.Tensor

    state_dict = checkpoint['model']
    if 'decoder.logits.bias' in state_dict:
        return checkpoint

    print('Fixing bias in output layer...')

    weight = state_dict['decoder.logits.weight']
    bias = weight.new_zeros([weight.size(0),], requires_grad=False)
    state_dict['decoder.logits.bias'] = bias
    if path:
        print('Fixed checkpoint saved!')
        torch.save(checkpoint, path)

    return checkpoint


def main(args):
    
    print("Loading checkpoints: " + " ".join(args.checkpoints))

    data = torch.load(args.checkpoints[0], map_location='cpu', )
    model_args = data['args']
    model_args.cpu = 'cuda' not in args.device
    model_args.context_embeddings_cache = args.device
    state = data['model']
    dictionary = Dictionary.load(DEFAULT_DICTIONARY)
    output_dictionary = ResourceManager.get_offsets_dictionary()

    target_manager = TargetManager(SequenceLabelingTaskKind.WSD)
    task = SequenceLabelingTask(model_args, dictionary, output_dictionary)

    if len(args.checkpoints) == 1:
        model = task.build_model(model_args).cpu().eval()
        model.load_state_dict(state, strict=True)
    else:
        checkpoints = LinearTaggerEnsembleModel.make_args_iterator(args.checkpoints)
        model = LinearTaggerEnsembleModel.build_model(
            checkpoints,
            task,
        )
        
    model = model.eval()
    model.to(args.device)

    datasets = []

    for corpus in args.xmls:
        if corpus.endswith('.data.xml'):
            dataset = WSDDataset.read_raganato(
                corpus,
                dictionary,
                use_synsets=True,
                max_length=args.max_length,
                on_error='keep',
                quiet=args.quiet,
                read_by=args.read_by,
            )
        else:
            with open(corpus, 'rb') as pkl:
                dataset = pickle.load(pkl)

        datasets.append(dataset)

    corpora = zip(args.xmls, datasets)

    for corpus, dataset in corpora:

        hit, tot = 0, 0
        all_answers = {}
        for sample_original in DataLoader(dataset, collate_fn=dataset.collater, batch_size=args.batch_size):
            with torch.no_grad():
                net_output = model(**{k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                                      for k, v in sample_original['net_input'].items()})
                lprobs = model.get_normalized_probs(net_output, log_probs=True).cpu()

            results, answers = target_manager.calulate_metrics(lprobs, sample_original)
            all_answers.update(answers)
            hit += results['hit']
            tot += results['tot']

        T = 0
        gold_answers = defaultdict(set)
        gold_path = Path(corpus.replace('data.xml', 'gold.key.txt'))
        bnids_map = None
        for line in gold_path.read_text().splitlines():
            pieces = line.strip().split(' ')
            if not pieces:
                continue
            trg, *gold = pieces
            T += 1
            for g in gold:
                if g.startswith('bn:'):
                    if bnids_map is None:
                        bnids_map = ResourceManager.get_bnids_to_offset_map()
                    o = bnids_map.get(g)
                    if o is None:
                        if args.on_error == 'keep':
                            o = {g,}
                            gold_answers[trg] |= o
                    else:
                        gold_answers[trg] |= o
                elif g.startswith('wn:'):
                    gold_answers[trg].add(g)
                else:
                    try:
                        o = make_offset(patched_lemma_from_key(g).synset())
                    except Exception:
                        o = None
                    if o is None:
                        if args.on_error == 'keep':
                            gold_answers[trg].add(g)
                    else:
                        gold_answers[trg].add(o)

        all_answers = {k: output_dictionary.symbols[v] for k, v in all_answers.items()}

        if args.on_error == 'skip':
            N = len([t for t, aa in gold_answers.items() if aa])
        else:
            N = len(gold_answers)
        ok, notok = 0, 0
        for k, answ in all_answers.items():
            gold = gold_answers.get(k)

            if not gold:
                continue
            if not answ or answ == '<unk>':
                continue
            if answ in gold:
                ok += 1
            else:
                notok += 1

        M = 0
        for k, gg in gold_answers.items():
            if args.on_error == 'skip' and (not gg):
                continue
            valid = False
            for g in gg:
                if g.startswith('wn:'):
                    valid = True
            if not valid:
                print(k, all_answers.get(k), gg)
            a = all_answers.get(k)
            if a is None or a == '<unk>':
                M += 1

        try:
            precision = ok / (ok + notok)
        except ZeroDivisionError:
            precision = 0.

        try:
            recall = ok / N
        except ZeroDivisionError:
            recall = 0.

        try:
            f1 = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.

        print(corpus)
        print(f'P: {precision}\tR: {recall}\tF1: {f1}\tN/T:{N}/{T}\tY/N/M/S: {ok}/{notok}/{M}/{T-N}')

        if args.predictions:
            if not os.path.exists(args.predictions):
                os.mkdir(args.predictions)
            name = ".".join(os.path.split(corpus)[-1].split('.')[:-2]) + '.results.key.txt'
            path = os.path.join(args.predictions, name)
            with open(path, 'w') as results_file:
                for k, v in sorted(all_answers.items()):
                    if not v or v == '<unk>':
                        v = ''
                    results_file.write(k + ' ' + v + '\n')


if __name__ == "__main__":

    parser = ArgumentParser(description="""
    WSD evaluation script. Predict and evaluate on corpora in the format of the WSD Framework of Raganato et al. (2017).
    """.strip())
    parser.add_argument(
        '-c', '--checkpoints', type=str, nargs='+', required=True,
        help='Path of trained EWISER checkpoint(s).')
    parser.add_argument(
        '-x', '--xmls', type=str, required=True, nargs='+',
        help='Raganato XML(s), <name>.gold.key.txt should be in the same folder. Multiple can be given.')
    parser.add_argument('-E', '--ensemble', action='store_true',
        help='Ensemble evaluation.')
    parser.add_argument(
        '-p', '--predictions', type=str, default='',
        help='Write predictions to this file.')
    parser.add_argument(
        '-l', '--max-length', type=int, default=100,
        help='Split input sequences in chunks of at most l=arg tokens.')
    parser.add_argument(
        '-b', '--batch-size', default=1, type=int,
        help='Batch size.')
    parser.add_argument(
        '-d', '--device', default='cpu',
        help='Device to use. (cpu, cuda, cuda:0 etc.)')
    parser.add_argument(
        '--read-by', default='text', choices=['sentence', 'text'],
        help='Read datasets by arg.')
    parser.add_argument(
        '--on-error', default='keep', choices=('keep', 'raise', 'skip'),
        help='What to do when some inconsisten instance is encountered.'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Do not print to stderr when some inconsistency is encountered.'
    )
    args = parser.parse_args()

    checkpoints = sorted(list(set(map(os.path.realpath, args.checkpoints))), key=lambda x: os.path.getmtime(x))
    if len(checkpoints) > 1 and not args.ensemble:
        for chkpt in checkpoints:
            args.checkpoints = [chkpt]
            main(args)
    else:
        args.checkpoints = checkpoints
        main(args)



