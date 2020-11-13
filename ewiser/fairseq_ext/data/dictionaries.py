import enum
import itertools
import logging
import math
import os
import pickle
from collections import defaultdict, OrderedDict, Counter
from typing import Set, List, Union, Dict, Tuple

import torch
from fairseq.data import Dictionary
from nltk.corpus import wordnet
import numpy as np

import ewiser
from ewiser.utils import EWISER_RES_DIR
from ewiser.fairseq_ext.data.utils import make_offset

DEFAULT_DICTIONARY = os.path.join(EWISER_RES_DIR, 'dictionaries', 'dict.txt')

class MFSManager:

    _INVALID_INTS = {0, 1, 2, 3}

    def __init__(
            self,
            seen_senses,
            seen_lemma_pos_to_sense_counter,
            lemma_pos_mask=False,
            reduce_mask_to_seen=True,
            threshold=1
    ) -> None:

        self.seen_senses = seen_senses
        self.seen_lemma_pos_to_sense_counter = seen_lemma_pos_to_sense_counter

        self.lemma_pos_mask = lemma_pos_mask
        self.reduce_mask_to_seen = reduce_mask_to_seen
        self.threshold = threshold

    @classmethod
    def from_lemma_pos_classes_couples(
            cls,
            lemma_pos_classes_couples: Dict[str, List[Tuple[Union[str, int], List[Union[str, int]]]]],
            *args, **kwargs
    ) -> 'MFSManager':
        """

        :param lemma_pos_classes_couples: has to be a dict where keys are langs and values are list of tuples
        where the first element is a lemma_pos in str form and the second is a list of offsets in str form
        :param lemma_pos_mask:
        :param reduce_mask_to_seen:
        """

        output_dictionary = ResourceManager.get_offsets_dictionary()

        def inner_defaultdict():
            return defaultdict(Counter)

        seen_senses = Counter()
        seen_lemma_pos_to_sense_counter = {}

        for lang, data in lemma_pos_classes_couples.items():

            for lemma_pos, senses in data:

                # lemma_pos = lemma_pos[:-2], lemma_pos[-1]
                if isinstance(lemma_pos, int):
                    lemma_pos_idx = lemma_pos
                else:
                    lemma_pos_idx = ResourceManager.get_lemma_pos_dictionary(lang).index(lemma_pos)

                senses = list(senses)
                if isinstance(senses[0], int):
                    senses_idx = senses
                else:
                    senses_idx = [output_dictionary.index(s) for s in senses]

                if not lang in seen_lemma_pos_to_sense_counter:
                    seen_lemma_pos_to_sense_counter[lang] = {}
                if not lemma_pos_idx in seen_lemma_pos_to_sense_counter[lang]:
                    seen_lemma_pos_to_sense_counter[lang][lemma_pos_idx] = Counter()
                else:
                    seen_lemma_pos_to_sense_counter[lang][lemma_pos_idx].update(senses_idx)
                seen_senses.update(senses_idx)

        return cls(seen_senses, seen_lemma_pos_to_sense_counter, *args, **kwargs)

    @classmethod
    def from_dataset(
            cls,
            dataset: 'WSDDataset',
            *args, **kwargs
    ) -> 'MFSManager':

        lemma_pos_classes_couples = defaultdict(list)

        for i in range(len(dataset)):
            sample = dataset[i]
            for j in range(sample['ntokens']):
                lemma_pos = int(sample['lemma_pos'][j])
                gold = sample['senses']['gold'][j]
                gold = [int(g) for g in gold]
                if gold[0] > 3:
                    lemma_pos_classes_couples[sample['lang']].append((lemma_pos, gold))

        return cls.from_lemma_pos_classes_couples(
            lemma_pos_classes_couples,
            *args, **kwargs
        )

    @classmethod
    def from_babelnet(
            cls,
            *args, **kwargs
    ):
        lemma_pos_classes_couples = defaultdict(list)
        for lang in cls.SUPPORTED_BABELNET_LANGUAGES:
            for lemma_pos_idx, _ in enumerate(ResourceManager.get_lemma_pos_dictionary(lang).symbols):
                senses = ResourceManager.get_lemma_pos_to_possible_offsets(lang)[lemma_pos_idx]
                for rank, sense in senses:
                    lemma_pos_classes_couples[lang].extend([lemma_pos_idx, [sense] * (len(senses) - rank)])

        inst = cls.from_lemma_pos_classes_couples(lemma_pos_classes_couples, *args, **kwargs)
        assert inst.lemma_pos_mask
        assert inst.reduce_mask_to_seen
        return inst

    def get_mask_indices(self, lemma_pos, lang='en'):

        if isinstance(lemma_pos, int):
            lemma_pos_idx = lemma_pos
        else:
            lemma_pos_idx = ResourceManager.get_lemma_pos_dictionary(lang)[lemma_pos]

        if self.lemma_pos_mask:
            return self._get_mask_indices_from_seen_lemma_pos(lemma_pos_idx, lang)
        else:
            return self._get_mask_indices_from_seen_senses(lemma_pos_idx, lang)

    @classmethod
    def _get_possible_senses(cls, lemma_pos_idx, lang='en'):
        return ResourceManager.get_lemma_pos_to_possible_offsets(lang)[lemma_pos_idx]

    def _get_mask_indices_from_seen_lemma_pos(self, lemma_pos_idx, lang='en'):
        if lemma_pos_idx not in self.seen_lemma_pos_to_sense_counter[lang]:
            return self._get_possible_senses(lemma_pos_idx, lang)[:1]
        elif self.reduce_mask_to_seen:
            poss = [s for s, c in self.seen_lemma_pos_to_sense_counter[lang][lemma_pos_idx].most_common() if c >= self.threshold]
            if not poss:
                poss = self._get_possible_senses(lemma_pos_idx, lang)[:1]
            return poss
        else:
            return self._get_possible_senses(lemma_pos_idx, lang)

    def _get_mask_indices_from_seen_senses(self, lemma_pos_idx, lang='en'):
        senses = self._get_possible_senses(lemma_pos_idx, lang)
        applicable_seen_senses = [s for s in senses if self.seen_senses[s] >= self.threshold]
        if applicable_seen_senses:
            return applicable_seen_senses
        else:
            return senses[:1]

    @classmethod
    def load(cls, path):

        with open(path, 'rb') as file:
            inst = pickle.load(file)

        assert isinstance(path, cls)

        return inst

class ResourceManager:

    _LEMMA2OFFSETS_SEP = "\t"

    _pos_dictionary = None
    _sensekeys_dictionary = None
    _offsets_dictionary = None
    _bnids_dictionary = None
    _lemma_pos_dictionary = {}
    _lemma_pos_to_possible_sensekeys = {}
    _lemma_pos_to_possible_offsets = {}
    _index_remap_offset_bnids = None
    _sensekeys_weights = None
    _offsets_weights = None

    @classmethod
    def get_pos_dictionary(cls) -> Dictionary:
        if cls._pos_dictionary is None:
            cls._pos_dictionary = Dictionary.load(os.path.join(EWISER_RES_DIR, 'dictionaries/pos.txt'))
        return cls._pos_dictionary

    @classmethod
    def get_sensekeys_dictionary(cls) -> Dictionary:
        if cls._sensekeys_dictionary is None:
            cls._sensekeys_dictionary = Dictionary.load(os.path.join(EWISER_RES_DIR, 'dictionaries/sensekeys.txt'))
        return cls._sensekeys_dictionary

    @classmethod
    def get_offsets_dictionary(cls) -> Dictionary:
        if cls._offsets_dictionary is None:
            cls._offsets_dictionary = Dictionary.load(os.path.join(EWISER_RES_DIR, 'dictionaries/offsets.txt'))
        return cls._offsets_dictionary

    @classmethod
    def get_offset_to_bnids_map(cls) -> Dict[str, str]:
        string_map = {}
        with open(os.path.join(EWISER_RES_DIR, 'dictionaries/bnids_map.txt')) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bnid, offset, *_ = line.split('\t')
                string_map[offset] = bnid
        return string_map

    @classmethod
    def get_bnids_to_offset_map(cls) -> Dict[str, Set[str]]:
        string_map = defaultdict(set)
        with open(os.path.join(EWISER_RES_DIR, 'dictionaries/bnids_map.txt')) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bnid, offset, *_ = line.split('\t')
                string_map[bnid].add(offset)
        return string_map

    @classmethod
    def get_bnids_dictionary(cls) -> Dictionary:
        if cls._bnids_dictionary is None:
            src_dictionary = cls.get_offsets_dictionary()
            tgt_dictionary = Dictionary()
            string_map = cls.get_offset_to_bnids_map()
            for idx, wn in enumerate(src_dictionary.symbols):
                if wn.startswith('wn:'):
                    tgt_dictionary.add_symbol(string_map[wn])
            tgt_dictionary.finalize()
            cls._bnids_dictionary = tgt_dictionary
        return cls._bnids_dictionary

    @classmethod
    def get_index_remap_offset_bnids(cls) -> List[int]:
        if cls._index_remap_offset_bnids is None:
            cls._index_remap_offset_bnids = []
            src_dictionary = cls.get_offsets_dictionary()
            tgt_dictionary = cls.get_bnids_dictionary()
            string_map = cls.get_offset_to_bnids_map()
            for src_idx, offset in enumerate(src_dictionary.symbols):
                if offset in string_map:
                    bnid = string_map[offset]
                    tgt_idx = tgt_dictionary.index(bnid)
                else:
                    tgt_idx = tgt_dictionary.index(offset)
                cls._index_remap_offset_bnids.append(tgt_idx)
        return cls._index_remap_offset_bnids

    @classmethod
    def get_senses_dictionary(cls, use_synsets=False):
        if use_synsets:
            return cls.get_offsets_dictionary()
        else:
            return cls.get_sensekeys_dictionary()

    @classmethod
    def get_lemma_pos_dictionary(cls, lang='en') -> Dictionary:
        if cls._lemma_pos_dictionary.get(lang) is None:
            cls._lemma_pos_dictionary[lang] = Dictionary.load(os.path.join(
                EWISER_RES_DIR, 'dictionaries', 'lemma_pos.' + lang + '.txt'
            ))
        return cls._lemma_pos_dictionary[lang]

    @classmethod
    def get_lemma_pos_to_possible_sensekeys(cls, lang='en') -> List[Set[int]]:
        assert lang == 'en'
        if cls._lemma_pos_to_possible_sensekeys.get(lang) is None:
            lemma_pos_dictionary = cls.get_lemma_pos_dictionary()
            sensekeys_dictionary = cls.get_sensekeys_dictionary()
            lemma_pos_to_possible_sensekeys = []
            for i, lemma_pos in enumerate(lemma_pos_dictionary.symbols):
                if i < lemma_pos_dictionary.nspecial:
                    lemma_pos_to_possible_sensekeys.append([lemma_pos_dictionary.index(lemma_pos)])
                else:
                    lemma, pos = lemma_pos.split('#')
                    senses = [sensekeys_dictionary.index(l.key()) for l in wordnet.lemmas(lemma, pos)]
                    lemma_pos_to_possible_sensekeys.append(senses)
            cls._lemma_pos_to_possible_sensekeys = lemma_pos_to_possible_sensekeys
        return cls._lemma_pos_to_possible_sensekeys[lang]

    @classmethod
    def get_lemma_pos_to_possible_offsets(cls, lang='en') -> List[List[int]]:
        if cls._lemma_pos_to_possible_offsets.get(lang) is None:
            offsets_dictionary = cls.get_offsets_dictionary()
            lemma_pos_to_possible_offsets = []
            if lang == 'en':
                lemma_pos_dictionary = cls.get_lemma_pos_dictionary(lang=lang)
                for i, lemma_pos in enumerate(lemma_pos_dictionary.symbols):
                    if i < lemma_pos_dictionary.nspecial:
                        lemma_pos_to_possible_offsets.append([lemma_pos_dictionary.index(lemma_pos)])
                    else:
                        lemma, pos = lemma_pos[:-2], lemma_pos[-1]
                        senses = [offsets_dictionary.index(make_offset(s.synset())) for s in wordnet.lemmas(lemma, pos)]
                        lemma_pos_to_possible_offsets.append(senses)
                        if lemma_pos not in lemma_pos_dictionary.indices:
                            raise KeyError(f'Lemma pos {lemma_pos} from the lemma pos to possible offsets dictionary'
                                           'is not in the lemma pos dictionary.')
            else:
                lemma_pos_dictionary = cls.get_lemma_pos_dictionary(lang)
                offsets_dictionary = cls.get_offsets_dictionary()
                string_map = cls.get_bnids_to_offset_map()

                lemma_pos_string_to_offsets_strings = {}

                with open(os.path.join(EWISER_RES_DIR, 'dictionaries', 'lemma_pos2offsets.' + lang + '.txt')) as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        lemma_pos, *bnids = line.split(cls._LEMMA2OFFSETS_SEP)
                        if lemma_pos not in lemma_pos_dictionary.indices:
                            raise KeyError(f'Lemma pos {lemma_pos} from the lemma pos to possible offsets dictionary'
                                           'is not in the lemma pos dictionary.')

                        if lemma_pos in lemma_pos_string_to_offsets_strings:
                            offsets = lemma_pos_string_to_offsets_strings[lemma_pos]
                        else:
                            offsets = []

                        offsets = offsets + [offset for bnid in bnids for offset in string_map[bnid]]
                        offsets = list(OrderedDict.fromkeys(offsets).keys())
                        lemma_pos_string_to_offsets_strings[lemma_pos] = offsets

                for i, lemma_pos in enumerate(lemma_pos_dictionary.symbols):
                    if lemma_pos not in lemma_pos_string_to_offsets_strings:
                        lemma_pos_to_possible_offsets.append([lemma_pos_dictionary.index(lemma_pos)])
                    else:
                        offsets = lemma_pos_string_to_offsets_strings[lemma_pos]
                        senses = [offsets_dictionary.index(o) for o in offsets]
                        lemma_pos_to_possible_offsets.append(senses)

            cls._lemma_pos_to_possible_offsets[lang] = lemma_pos_to_possible_offsets
        return cls._lemma_pos_to_possible_offsets[lang]

    @classmethod
    def get_lemma_pos_to_possible_senses(cls, use_synsets=False) -> List[Set[int]]:
        if use_synsets:
            return cls.get_lemma_pos_to_possible_offsets()
        else:
            return cls.get_lemma_pos_to_possible_sensekeys()

    @classmethod
    def get_sensekey_weights(cls):
        if cls._sensekeys_weights is None:
            weights = []
            for s in cls.get_sensekeys_dictionary().symbols:
                if s.startswith("<"):
                    weights.append(0.0)
                else:
                    weights.append(1.0)
            cls._sensekeys_weights = np.array(weights)
        return cls

    @classmethod
    def make_adjacency_from_files(cls, *paths, input_keys=None, symmetric=True, max_incoming=5, filter_out_tops=False):

        self_loops_count = 0
        tops_count = 0
        offsets = cls.get_offsets_dictionary()

        # #peek and infer
        # if input_keys is None:
        #     with open(paths[0]) as file:
        #         for line in file:
        #             line = line.strip()
        #             if not line:
        #                 continue
        #             else:
        #                 offset1, offset2, *info = line.split()
        #                 if offset1.startswith('bn:'):
        #                     input_keys = 'bnids'
        #                 elif offset1.startswith('wn:'):
        #                     input_keys = 'offsets'
        #                 else:
        #                     input_keys = 'sensekeys'
        #                 break
        # assert input_keys is not None

        remap = cls.get_bnids_to_offset_map()

        import networkx as nx
        from nltk.corpus import wordnet

        g = nx.DiGraph()

        size = torch.Size([len(offsets)] * 2)

        for path in paths:

            with open(path) as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    offset1, offset2, *info = line.split('\t')
                    if 0 <= len(info) <= 1:
                        w = None
                    else:
                        try:
                            w = float(info[1])
                        except ValueError:
                            w = None

                    if offset1.startswith('bn:'):
                        offsets1 = remap.get(offset1)
                    elif offset1.startswith('wn:'):
                        offsets1 = [offset1]
                    else:
                        raise NotImplementedError

                    if offset2.startswith('bn:'):
                        offsets2 = remap.get(offset2)
                    elif offset2.startswith('wn:'):
                        offsets2 = [offset2]
                    else:
                        raise NotImplementedError

                    for offset1, offset2 in itertools.product(offsets1, offsets2):
                        offset1 = fix_offset(offset1) # v -> child in hypernymy
                        offset2 = fix_offset(offset2) # u -> father in hypernymy
                        syn1 = wordnet.synset_from_pos_and_offset(offset1[-1], int(offset1[3:-1]))
                        syn2 = wordnet.synset_from_pos_and_offset(offset2[-1], int(offset2[3:-1]))

                        if filter_out_tops:
                            if syn1.lexname().lower().endswith('tops'):
                                if syn1.hyponyms() and syn1.hyponyms()[0].lexname().lower().endswith('tops'):
                                    tops_count += 1
                                    continue
                            if syn2.lexname().lower().endswith('tops'):
                                if syn2.hyponyms() and syn2.hyponyms()[0].lexname().lower().endswith('tops'):
                                    tops_count += 1
                                    continue
                        
                        trg_node = offsets.index(offset1)
                        src_node = offsets.index(offset2)
                        if src_node != trg_node:
                            g.add_edge(src_node, trg_node, w=w)
                        else:
                            self_loops_count += 1

        if self_loops_count > 0:
            logging.warning(f'{self_loops_count} self loops in the adjacency list files, which have been ignored!')
        if filter_out_tops and tops_count > 0:
            logging.warning(f'{tops_count} edges with either source or target in noun.Tops, which have been ignored!')
        coordinates = []
        values = []

        # TODO: remove this debug mess
        # import os
        # TRANSPOSE = os.environ['STRUCTURED_LOGITS_TRANSPOSE']
        # DEGREE =    os.environ['STRUCTURED_LOGITS_DEGREE']
        # WHICH =     os.environ['STRUCTURED_LOGITS_WHICH']
        # assert TRANSPOSE in ('y', 'n')
        # assert DEGREE in ('i', 'o')
        # assert WHICH in ('src', 'trg')

        for trg in g.nodes():
            src_nodes = list(g.predecessors(trg))
            weights = [g.get_edge_data(src, trg).get('w') for src in src_nodes]
            weights = [1. if w is None else w for w in weights]
            # if w is None:
            #     exp1 = 0.
            #     exp2 = 1.
            #     degree = (
            #         (g.out_degree(src_node)  ** exp1) *
            #         (g.in_degree(trg_node)   ** exp2)
            #     ) ** (1 / (exp1 + exp2))
            #     w = 1./ degree
            s = sum(weights)
            weights = [w / s for w in weights]
            for src, w in sorted(zip(src_nodes, weights), key=lambda x:x[1], reverse=True)[:max_incoming]:
                coordinates.append((trg, src)) # REVERSED!!!
                values.append(w)

        coordinates = torch.LongTensor(coordinates).t()
        values = torch.FloatTensor(values)
        adjacency = torch.sparse.FloatTensor(coordinates, values, size).coalesce()

        # mouse = wordnet.synset('mouse.n.01')
        # dad = mouse.hypernyms()[0]
        # kid = mouse.hyponyms()[0]
        #
        # mouse_idx = offsets.index('wn:' + str(mouse.offset()).zfill(8) + 'n')
        # dad_idx = offsets.index('wn:' + str(dad.offset()).zfill(8) + 'n')
        # kid_idx = offsets.index('wn:' + str(kid.offset()).zfill(8) + 'n')
        #
        # assert adjacency[mouse_idx, dad_idx] > 0
        # assert adjacency[dad_idx, mouse_idx] == 0
        # assert adjacency[mouse_idx, kid_idx] == 0

        return adjacency


class SequenceLabelingTaskKind(enum.Enum):

    WSD = 0
    POS = 1

class TargetManager:

    def __init__(self, kind: Union[str, SequenceLabelingTaskKind], **other_stuff):
        if isinstance(kind, str):
            self.kind = SequenceLabelingTaskKind[kind.upper()]
        else:
            self.kind = kind
        assert isinstance(self.kind, SequenceLabelingTaskKind)
        self._setup(other_stuff)

    def _setup(self, other_stuff):
        getattr(self, '_setup_' + self.kind.name, lambda other_stuff: None)(other_stuff)

    def calulate_metrics(self, lprobs, sample):
        return getattr(self, '_calculate_metrics_' + self.kind.name, self._calculate_metrics_DEFAULT)(lprobs, sample)

    def get_targets(self, sample):
        return getattr(self, '_get_targets_' + self.kind.name)(sample)

    def _setup_WSD(self, other_stuff):
        if other_stuff.get('mfs'):
            print('USING MFS!')
            assert isinstance(other_stuff['mfs'], MFSManager)
            self.mfs_manager = other_stuff['mfs']
        else:
            self.mfs_manager = None
        self.only_calc_on_ids = other_stuff.get('only_calc_on_ids', True)

    def _calculate_metrics_DEFAULT(self, lprobs, sample):
        preds = lprobs.view(-1, lprobs.size(-1)).argmax(1)
        true = self.get_targets(sample).view(-1)
        return {'hit': (preds == true).sum().item(), 'tot': len(true)}, preds.detach()

    def _calculate_metrics_WSD(self, lprobs, sample):
        senses = sample['senses']
        answers = {}
        hit = 0
        tot = 0
        for i, (ids, all_, gold, lemma_pos, lang) in enumerate(
                zip(senses['ids'], senses['all'], senses['gold'], sample['lemma_pos'], sample['net_input']['langs'])
        ):
            if self.only_calc_on_ids:
                js = [j for j, id_ in enumerate(ids) if id_]
            else:
                js = list(range(len(ids)))
            for j in js:
                if self.mfs_manager:
                    poss = self.mfs_manager.get_mask_indices(lemma_pos[j].item(), lang)
                else:
                    poss = list(all_[j])
                try:
                    pred = poss[lprobs[i][j][poss].argmax().item()]
                except TypeError as e:
                    print(all_)
                    print(ids[j])
                    print(gold[j])
                    print(poss)
                    print(i)
                    print(j)
                    raise e
                tot += 1

                if len(gold[j]) == 1 and gold[j][0] == 3:
                    pass
                elif pred in gold[j]:
                    hit += 1
                if self.only_calc_on_ids:
                    answers[ids[j]] = pred
                if ResourceManager.get_offsets_dictionary().symbols[pred] == '<unk>':
                    pass
        return {'hit': hit, 'tot': tot}, answers

    def _get_targets_DEFAULT(self, sample):
        return sample['target']

    def _get_targets_WSD(self, sample):
        return sample['senses']['target']

    def _get_targets_POS(self, sample):
        return sample['pos']

    def _get_adjacency_WSD(self):
        offsets = ResourceManager.get_offsets_dictionary()

        coordinates = []
        values = []
        size = torch.Size([len(offsets)] * 2)

        for i, offset1 in enumerate(offsets.symbols):
            if offset1.startswith('wn:'):
                synset1 = wordnet.synset_from_pos_and_offset(offset1[-1], int(offset1[3:-1]))
                for synset2 in itertools.chain(
                        synset1.hypernyms(),
                        synset1.hyponyms(),
                        synset1.similar_tos(),
                ):
                    offset2 = make_offset(synset2)
                    j = offsets.index(offset2)
                    coordinates.extend([(i, j), (j, i)])
                    values.extend([1., 1.])

        coordinates = torch.LongTensor(sorted(coordinates)).t()
        values = torch.FloatTensor(values)
        adjacency = torch.sparse.FloatTensor(coordinates, values, size)
        return adjacency


    @classmethod
    def slice_batch(cls, batch, slice):
        assert isinstance(batch, dict)
        sliced = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced[k] = cls.slice_batch(v, slice)
            elif isinstance(v, torch.Tensor):
                sliced[k] = v[:, slice]
            elif isinstance(v, list):
                sliced[k] = [inner[slice] for inner in v]
            else:
                raise TypeError
        return sliced

    @staticmethod
    def select_flat_tensor(tensor, mask):
        tensor_flat = tensor.view(-1, *tensor.shape[2:])
        return tensor_flat[mask].unsqueeze(0)

    @staticmethod
    def select_flat_list_of_lists(lol, mask):
        list_ = []
        k = 0
        for l in lol:
            for elem in l:
                if mask[k]:
                    list_.append(elem)
                k += 1
        return [list_]

    @classmethod
    def remove_non_targets(cls, batch, nspecial):

        target_tensor: torch.LongTensor = batch['senses']['target']

        assert isinstance(batch, dict)
        flat_mask = (target_tensor > nspecial).view(-1)
        src_lengths = flat_mask.long().sum().unsqueeze(0)

        batch['ntokens'] = src_lengths.sum().item()
        batch['lemma_pos'] = cls.select_flat_tensor(batch['lemma_pos'], flat_mask)

        net_input = batch['net_input']
        net_input['src_tokens'] = cls.select_flat_tensor(net_input['src_tokens'], flat_mask)
        net_input['src_lengths'] = src_lengths
        net_input['src_tokens_str'] = cls.select_flat_list_of_lists(net_input['src_tokens_str'], flat_mask)
        net_input['langs'] = net_input['langs'][:1]
        if 'cached_vectors' in net_input:
            net_input['cached_vectors'] = cls.select_flat_tensor(net_input['cached_vectors'], flat_mask)

        senses = batch['senses']
        senses['target'] = cls.select_flat_tensor(senses['target'], flat_mask)
        senses['gold'] = cls.select_flat_list_of_lists(senses['gold'], flat_mask)
        senses['all'] = cls.select_flat_list_of_lists(senses['all'], flat_mask)
        senses['ids'] = cls.select_flat_list_of_lists(senses['ids'], flat_mask)

        return batch


def fix_offset(offset):
    offsets = ResourceManager.get_offsets_dictionary()
    if offset.endswith('a'):
        try:
            offsets.indices[offset]
            return offset
        except KeyError as e:
            fixed_offset = offset[:-1] + 's'
            if fixed_offset not in offsets.indices:
                raise e
            else:
                return fixed_offset
    else:
        offsets.indices[offset]
        return offset
