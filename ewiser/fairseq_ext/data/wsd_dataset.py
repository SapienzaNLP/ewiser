import bisect
import enum
import itertools
import json
import logging
import math
import multiprocessing
import os
import pickle
import random
import tempfile
import unicodedata
import warnings
from collections import namedtuple, OrderedDict

import h5py
from typing import List, Set, Tuple, Union, Dict, Optional
from xml.etree import cElementTree as etree
from xml.etree.cElementTree import parse
from joblib import Parallel, delayed

from fairseq.data import FairseqDataset, Dictionary
from nltk.corpus import wordnet
import numpy as np
import torch
from tqdm import tqdm

from ewiser.fairseq_ext.data.dictionaries import ResourceManager, DEFAULT_DICTIONARY, TargetManager
from ewiser.fairseq_ext.data.utils import _longest, make_offset, patched_lemma_from_key
from ewiser.fairseq_ext.modules.contextual_embeddings import BaseContextualEmbedder, BERTEmbedder
from ewiser.fairseq_ext.utils import remove_dup

nlp = None
sent_tokenizer = None
nlp_stanford = None

def read_by_text_iter(xml_path):

    i = 0
    it = etree.iterparse(xml_path, events=('start', 'end'))
    _, root = next(it)
    for event, elem in it:
        if event == 'end' and elem.tag == 'text':
            for sentence in elem:
                for word in sentence:
                    yield i, word
            i += 1
        root.clear()


def read_by_sent_iter(xml_path):
    i = 0
    it = etree.iterparse(xml_path, events=('start', 'end'))
    _, root = next(it)
    for event, elem in it:
        if event == 'end' and elem.tag == 'sentence':
            for word in elem:
                yield i, word
            i += 1
        root.clear()


def _read_by_text(xml):

    for i, text in enumerate(xml.iter("text")):
        for sentence in text:
            for word in sentence:
                yield i, word

def _read_by_sent(xml):
    for i, sent in enumerate(xml.iter("sentence")):
        for word in sent:
            yield i, word

class RaganatoReadBy(enum.Enum):

    SENTENCE = read_by_sent_iter
    TEXT = read_by_text_iter

def _read_raganato_xml(
        xml_path: str,
        read_by: Union[str, RaganatoReadBy] = RaganatoReadBy.TEXT,
        dictionary: Optional[Dictionary] = None,
        tagset='universal', #universal, multilingual
        lang='en',
        inst_to_keep=None,
        on_error='skip', # skip, keep, raise
        quiet=True,
) -> Tuple[np.ndarray, List[str], Dict[int, str]]:

    if isinstance(read_by, str):
        read_by = getattr(RaganatoReadBy, read_by.upper())
    elif isinstance(read_by, RaganatoReadBy):
        read_by = read_by.value
    else:
        raise TypeError

    if not dictionary:
        dictionary = Dictionary.load(DEFAULT_DICTIONARY)

    assert tagset in {'universal', 'multilingual'}

    oov_dictionary = {}

    pos_dictionary = ResourceManager.get_pos_dictionary()
    lemma_pos_dictionary = ResourceManager.get_lemma_pos_dictionary(lang)
    lemma_pos_to_possible_offsets_map = ResourceManager.get_lemma_pos_to_possible_offsets(lang)
    offsets_dictionary = ResourceManager.get_offsets_dictionary()

    total_text_n = []
    total_token = []
    total_lemma_pos = []
    total_pos = []
    total_gold_indices = []

    token = []
    token_strings = []
    lemma_pos = []
    pos = []
    gold_indices = []
    target_labels = []
    gold_idx = 0

    def discharge():

        cond1 = bool(token)
        # cond2 = True
        cond2 = any(g > -1 for g in gold_indices)

        if cond1 and cond2:

            if not total_text_n:
                old_text_n = -1
            else:
                old_text_n = total_text_n[-1]

            text_n = [old_text_n + 1] * len(token)

            for token_number, (t, ts) in enumerate(zip(token, token_strings), start=len(total_token)):

                if t == dictionary.unk_index:
                    oov_dictionary[token_number] = ts

            total_text_n.extend(text_n)
            total_token.extend(token)
            total_lemma_pos.extend(lemma_pos)
            total_pos.extend(pos)
            total_gold_indices.extend(gold_indices)

        token.clear()
        token_strings.clear()
        lemma_pos.clear()
        pos.clear()
        gold_indices.clear()

    old_text_number = -1

    for token_number, (text_number, word) in enumerate(read_by(xml_path)):

        if not word.text:
            continue

        if text_number != old_text_number:
            discharge()

        old_text_number = text_number

        if tagset == 'universal':
            t = word.text.replace('_', ' ')
            p = word.attrib["pos"]
            if len(p) == 1:
                lp = lemma_pos_dictionary.index(
                    word.attrib.get("lemma", 'X').lower() + '#' + p.lower().replace('j', 'a')
                )
            else:
                lp = lemma_pos_dictionary.index(
                    word.attrib.get("lemma", 'X').lower() + '#' + _ud_to_wn.get(p, 'n')
                )
            pos.append(pos_dictionary.index(p))

        elif tagset == 'multilingual':
            raise
            try:
                text = word.text.strip().replace(' ', '_')
            except Exception as e:
                print(etree.tostring(word))
                raise e
            try:
                t, l, p = text.split('/')
            except ValueError as e:
                t = text.split('/')[0]
                p = word.attrib["pos"][0].lower()
                l = word.attrib["lemma"].lower()
                print(etree.tostring(word))
            l = l.lower().strip()
            p = p.lower().strip()
            lp = lemma_pos_dictionary.index(l + '#' + p)
            pos.append(pos_dictionary.unk_index)

        else:
            raise

        token_strings.append(t)
        token.append(dictionary.index(t))
        lemma_pos.append(lp)

        idx = word.attrib.get("id")
        if idx and word.tag == 'instance':

            ignore = False
            in_lemma_pos = True

            if len(p) == 1:
                lp_string = unicodedata.normalize('NFC', word.attrib["lemma"].lower()) + '#' + word.attrib['pos'].lower().replace('j', 'a')
            else:
                lem = word.attrib.get("lemma").lower()
                wnpos = _ud_to_wn.get(word.attrib["pos"], 'n')
                if not lem and word.tag == 'instance':
                    lem = wordnet.morphy(t, wnpos)
                    if not lem:
                        lem = ''
                lp = lemma_pos_dictionary.index(
                    lem + '#' + wnpos
                )
                lp_string = unicodedata.normalize('NFC', lem) + '#' + wnpos
            if lp_string not in lemma_pos_dictionary.indices:
                in_lemma_pos = False
                msg = f'Lemma and pos "{lp_string}" for instance "{idx}" not in the lemma pos dictionary.'
                if on_error == 'skip':
                    ignore = True
                    if not quiet:
                        logging.warning('SKIP:' + msg)
                elif on_error == 'keep':
                    if not quiet:
                        logging.warning('KEEP:' + msg)
                else:
                    raise KeyError(msg)

            if inst_to_keep and (idx in inst_to_keep) and in_lemma_pos:

                gold = inst_to_keep[idx]
                possible = lemma_pos_to_possible_offsets_map[lp]
                possible_str = [offsets_dictionary.symbols[x] for x in possible]
                for g in gold:

                    o = offsets_dictionary.symbols[g]

                    if o not in possible_str:

                        msg = (
                            f'"{o}" (instance "{idx}") '
                            f'not among the possible for lemma pos "{lp_string}". '
                            f'Possible: {possible_str}.'
                        )
                        if on_error == 'skip':
                            ignore = True
                            if not quiet:
                                logging.warning('SKIP:' + msg)
                        elif on_error == 'keep':
                            if not quiet:
                                logging.warning('KEEP:' + msg)
                        else:
                            raise KeyError(msg)

            if ignore:
                gold_indices.append(-1)

            elif (not inst_to_keep) or (idx in inst_to_keep):

                target_labels.append(idx)
                gold_indices.append(gold_idx)
                gold_idx += 1

            else:
                gold_indices.append(-1)
        else:
            gold_indices.append(-1)

    discharge()

    text_n = np.array(total_text_n, dtype=np.int64)
    token = np.array(total_token, dtype=np.int32)
    lemma_pos = np.array(total_lemma_pos, dtype=np.int32)
    pos = np.array(total_pos, dtype=np.int8)
    gold_indices = np.array(total_gold_indices, dtype=np.int32)

    raw_data = np.rec.fromarrays(
        [text_n, token, lemma_pos, pos, gold_indices],
        names=['text_n', 'token', 'lemma_pos', 'pos', 'gold_indices']
    )

    return raw_data, target_labels, oov_dictionary

def _read_raganato_gold_(
        gold_path: str,
        _use_synsets: bool = False,
        input_keys: str = "sensekeys",
        on_error: str = "skip", # skip, keep, raise
        quiet: bool = False,
) -> Dict[str, List[int]]:

    if input_keys == 'bnids':
        bnids_map = ResourceManager.get_bnids_to_offset_map()

    target_dict = {}
    dictionary = \
        ResourceManager.get_offsets_dictionary() if _use_synsets else ResourceManager.get_sensekeys_dictionary()
    with open(gold_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if input_keys == 'sensekeys':
                instance, *sensekeys = line.split()
                try:
                    senses = [make_offset(patched_lemma_from_key(sk, wordnet).synset()) for sk in sensekeys]
                except Exception as e:
                    print(instance, sensekeys)
                    raise e
            elif input_keys == 'offsets':
                instance, *offsets = line.split()
                offsets_ = []
                for offset in offsets:
                    if offset not in dictionary.indices:
                        msg = f'Error in gold file for instance {instance}: {offset} is not valid.'
                        if on_error == 'keep':
                            offsets_.append(offset)
                            if not quiet:
                                logging.warning('KEEP: ' + msg)
                        elif on_error == 'skip':
                            if not quiet:
                                logging.warning('SKIP: ' + msg)
                        else:
                            raise KeyError(msg)
                    else:
                        offsets_.append(offset)
                senses = offsets_
            elif input_keys == 'bnids':
                instance, *bnids = line.split()
                bnids_ = []
                for bnid in bnids:
                    if bnid not in bnids_map:
                        msg = f'Error in gold file for instance {instance}: {bnid} is not valid or not in WordNet subgraph.'
                        if on_error == 'keep':
                            bnids_.append(bnid)
                            if not quiet:
                                logging.warning('KEEP: ' + msg)
                        elif on_error == 'skip':
                            if not quiet:
                                logging.warning('SKIP: ' + msg)
                        else:
                            raise KeyError(msg)
                    else:
                        bnids_.append(bnid)
                bnids = bnids_
                senses = list({s for b in bnids for s in bnids_map[b]})
            else:
                senses = sensekeys

            if senses:
                senses = [dictionary.index(s) for s in senses]
                senses = remove_dup(senses, dictionary)
                target_dict[instance] = senses
            elif on_error == 'skip':
                if not quiet:
                    logging.warning(f'SKIP: empty gold for instance {instance}.')
            elif on_error == 'keep':
                target_dict[instance] = senses
                if not quiet:
                    logging.warning(f'KEEP: empty gold for instance {instance}.')
            else:
                raise ValueError(f'empty gold for instance {instance}.')
    return target_dict

Tok = namedtuple('Tok', ["text", "attrib"])

def _parse_with_spacy(path, threads=-1):
    global nlp
    global sent_tokenizer
    if nlp is None:
        import spacy
        nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
    if sent_tokenizer is None:
        import nltk
        sent_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    with open(path) as file:
        lines = [line.strip() for line in file]

    lines = (line for line in lines if line)

    lines = Parallel(threads)(delayed(sent_tokenizer.tokenize)(line) for line in lines)
    sents = (sent for line in lines for sent in line)

    sents = nlp.pipe(sents, batch_size=100, n_threads=threads)

    for i, sent in enumerate(sents):
        for tok in sent:
            attrib = {'pos': tok.pos_, 'lemma': tok.lemma_}
            tok = Tok(tok.text, attrib)
            yield i, tok

def _parse_with_stanford(path, **kwargs):
    global nlp_stanford
    if not nlp_stanford:
        import stanfordnlp
        nlp_stanford = stanfordnlp.Pipeline(processors='tokenize,pos,lemma', treebank='en_ewt', **kwargs)
    with open(path) as file:
        sents = nlp_stanford(file.read().strip()).sentences

    for i, sent in enumerate(sents):
        for tok in sent:
            attrib = {'pos': tok.upos, 'lemma': tok.lemma}
            tok = Tok(tok.text, attrib)
            yield i, tok


def _read_plaintext(plaintext_path: str, dictionary=None, merge_with_prob: float=0.0) -> Tuple[np.ndarray, List[str], Dict[int, str]]:
    assert dictionary is not None
    oov_dictionary = {}

    pos_dictionary = ResourceManager.get_pos_dictionary()
    lemma_pos_dictionary = ResourceManager.get_lemma_pos_dictionary()

    text_n = []
    token = []
    lemma_pos = []
    pos = []
    gold_indices = []
    target_labels = []
    gold_idx = 0

    text_i = 0
    old_text_n_real = 0
    for t_n, (text_n_real, word) in enumerate(_parse_with_spacy(plaintext_path)):

        if merge_with_prob <= 0.:
            text_i = text_n_real
        else:
            if text_n_real == 0:
                pass
            elif old_text_n_real != text_n_real:
                if random.random() > merge_with_prob:
                    pass
                else:
                    text_i += 1
                old_text_n_real = text_n_real

        text_n.append(text_i)
        t = word.text.replace(' ', '_')
        if t not in dictionary.indices:
            oov_dictionary[t_n] = t
            t = _longest(t)
        token.append(dictionary.index(t))
        p = word.attrib["pos"]
        lp = lemma_pos_dictionary.index(word.attrib["lemma"].lower() + '#' + _ud_to_wn.get(p, 'x'))
        lemma_pos.append(lp)
        pos.append(pos_dictionary.index(p))
        idx = word.attrib.get("id")
        if idx:
            target_labels.append(idx)
            gold_indices.append(gold_idx)
            gold_idx += 1

        else:
            gold_indices.append(-1)

    text_n = np.array(text_n, dtype=np.int64)
    token = np.array(token, dtype=np.int32)
    lemma_pos = np.array(lemma_pos, dtype=np.int32)
    pos = np.array(pos, dtype=np.int8)
    gold_indices = np.array(gold_indices, dtype=np.int32)

    raw_data = np.rec.fromarrays(
        [text_n, token, lemma_pos, pos, gold_indices],
        names=['text_n', 'token', 'lemma_pos', 'pos', 'gold_indices']
    )

    return raw_data, target_labels, oov_dictionary


class WSDDatasetBuilder:

    def __init__(self, path, dictionary=None, use_synsets=True, keep_string_data=False, lang="en"):

        assert use_synsets

        if not dictionary:

            dictionary = Dictionary.load(DEFAULT_DICTIONARY)

        if not os.path.exists(path):
            os.makedirs(path)

        self.vectors_path = os.path.join(path, 'vectors.hdf5')
        self.gold_path = os.path.join(path, 'gold.pkl')
        if os.path.exists(self.vectors_path):
            os.remove(self.vectors_path)
        if os.path.exists(self.gold_path):
            os.remove(self.gold_path)

        self.keep_string_data = keep_string_data
        if keep_string_data:
            self.oov_dictionary_path = os.path.join(path, 'oov.pkl')
            if os.path.exists(self.oov_dictionary_path):
                os.remove(self.oov_dictionary_path)
            self.oov_dictionary = {}
        else:
            self.oov_dictionary = None

        self.metadata = {
            "lang": lang
        }
        self.metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        self.lang = self.metadata['lang']

        self.hd5_file = h5py.File(self.vectors_path, mode='w')
        self.token_data = None
        self.seq_data = None
        self.gold = []

        self._max_gold = 0
        self._max_sequence = 0

        self.dictionary = dictionary
        self.output_dictionary = ResourceManager.get_offsets_dictionary()

        self.use_synsets = use_synsets

    def __del__(self):
        if self.hd5_file is not None:
            self.hd5_file.close()

    def finalize(self):
        self.hd5_file.close()
        self.hd5_file = None
        with open(self.gold_path, 'wb') as pkl:
            pickle.dump(self.gold, pkl)
        if self.oov_dictionary is not None:
            with open(self.oov_dictionary_path, 'wb') as pkl:
                pickle.dump(self.oov_dictionary, pkl)
        with open(self.metadata_path, 'w') as json_hdl:
            json.dump(self.metadata, json_hdl)

    def add_plaintext(self, plaintext_path, max_length=100, merge_with_prob=0.0):
        raw_data, _, oov_dictionary = _read_plaintext(plaintext_path, dictionary=self.dictionary, merge_with_prob=merge_with_prob)
        gold = []
        self._add_corpus(raw_data, gold, max_length=max_length, oov_dictionary=oov_dictionary)

    def add_plaintexts(self, plaintext_paths, max_length=100, n_jobs=1, merge_with_prob=0.0):
        args = zip(plaintext_paths, itertools.repeat(self.dictionary), itertools.repeat(merge_with_prob))
        with multiprocessing.Pool(n_jobs) as pool:
            for raw_data, _, oov_dictionary in pool.starmap(_read_plaintext, args):
                gold = []
                self._add_corpus(raw_data, gold, max_length=max_length, oov_dictionary=oov_dictionary)

    def add_raganato(
            self,
            xml_path,
            max_length=100,
            input_keys='sensekeys',
            tagset='universal',
            on_error='skip',
            quiet=False,
            read_by=RaganatoReadBy.TEXT,
    ):
        
        target_dict = _read_raganato_gold_(
            xml_path.replace('.data.xml', '.gold.key.txt'),
            _use_synsets=True,
            input_keys=input_keys,
            on_error=on_error,
            quiet=quiet,
        )
        
        raw_data, target_labels, oov_dictionary = _read_raganato_xml(
            xml_path,
            dictionary=self.dictionary,
            tagset=tagset,
            lang=self.lang,
            inst_to_keep=target_dict,
            on_error=on_error,
            quiet=quiet,
            read_by=read_by,
        )

        if len(target_dict) == len(target_labels):
            gold = [(lab, target_dict[lab]) for lab in target_labels]

        else: #some keys are not present
            gold = []
            for i in range(len(raw_data)):
                g_idx = raw_data['gold_indices'][i]
                if g_idx > -1:
                    lab = target_labels[g_idx]
                    g = target_dict.get(lab)
                    if g:
                        gold.append((lab, g))
                    else:
                        raw_data['gold_indices'][i] = -1

        self._add_corpus(raw_data, gold, max_length=max_length, oov_dictionary=oov_dictionary)

    def _add_corpus(self, raw_data: np.ndarray, gold: List[Tuple[str, List[int]]], max_length=100, oov_dictionary=None):

        if self.token_data is None:
            start = 0
            end = raw_data.shape[0]
            self.token_data = \
                self.hd5_file.create_dataset('token_data', shape=raw_data.shape, dtype=raw_data.dtype, maxshape=(None,))
            if oov_dictionary is not None and self.oov_dictionary is not None:
                oov_dictionary = {i + start: w for i, w in oov_dictionary.items()}
                self.oov_dictionary.update(oov_dictionary)

        else:
            start = self.token_data.shape[0]
            end = start + raw_data.shape[0]
            if oov_dictionary is not None and self.oov_dictionary is not None:
                oov_dictionary = {i + start: w for i, w in oov_dictionary.items()}
                self.oov_dictionary.update(oov_dictionary)
            self.token_data.resize(size=(self.token_data.shape[0] + raw_data.shape[0],))

        # fixing offsets
        n_gold = len(gold)
        raw_data['text_n'] += self._max_sequence
        raw_data['gold_indices'][raw_data['gold_indices'] != -1] = \
            np.arange(n_gold, dtype=raw_data['gold_indices'].dtype) + self._max_gold
        self._max_sequence = raw_data['text_n'][-1]
        self._max_gold += n_gold

        self.gold.extend(gold)

        self.token_data[start:end] = raw_data

        seq_raw_data = []
        start_seq = start
        # for _, seq in pd.Series(raw_data['text_n']).groupby(lambda x: x):
        #     seq = seq.values
        text_n = raw_data['text_n']
        for seq in np.split(text_n, np.unique(text_n, return_index=True)[1][1:]):
            n_chunks = math.ceil(len(seq) / max_length)
            for seq_chunk in np.array_split(seq, n_chunks):
                seq_raw_data.append([start_seq, len(seq_chunk)])
                start_seq += len(seq_chunk)

        seq_raw_data = np.array(seq_raw_data)
        if self.seq_data is None:
            start = 0
            end = seq_raw_data.shape[0]
            self.seq_data = \
                self.hd5_file.create_dataset('seq_data', shape=seq_raw_data.shape, dtype=seq_raw_data.dtype, maxshape=(None, 2))
        else:
            start = self.seq_data.shape[0]
            end = start + seq_raw_data.shape[0]
            self.seq_data.resize(size=(self.seq_data.shape[0] + seq_raw_data.shape[0], seq_raw_data.shape[1]))

        self.seq_data[start:end] = seq_raw_data

class WSDDataset(FairseqDataset):

    #Data
    token_data: h5py.Dataset
    seq_data: h5py.Dataset
    gold: List[Set[int]]
    sizes: np.array
    lang: str

    #Dictionaries
    dictionary: Dictionary
    output_dictionary: Dictionary
    pos_dictionary: Dictionary
    lemma_pos_dictionary: Dictionary
    lemma_pos_to_possible_senses: List[Set[int]]

    #Parameters
    shuffle: bool
    add_monosemous: bool
    use_synsets: bool

    @classmethod
    def read_raganato(
            cls,
            xml_path,
            dictionary=None,
            use_synsets=True,
            add_monosemous=False,
            max_length=100,
            input_keys=None,
            lang=None,
            tagset='universal',
            lazy=False,
            cache_path=None,
            on_error='keep',
            quiet=True,
            read_by=RaganatoReadBy.TEXT,
    ):
        #detect lang
        lang = lang or next(etree.iterparse(xml_path, events=('start',)))[1].attrib.get('lang', 'en')


        #detect input
        if not input_keys:
            with open(xml_path.replace('.data.xml', '.gold.key.txt')) as file:
                for line in file:
                    line = line.strip()
                    if not line or 'unk' in line.lower():
                        continue
                    if 'bn:' in line:
                        input_keys = 'bnids'
                    elif 'wn:' in line:
                        input_keys = 'offsets'
                    else:
                        input_keys = 'sensekeys'

        if cache_path:
            from_tmp = False
        else:
            cache_path = tempfile.mkdtemp('ewiser-datasets')
            from_tmp = True
        builder = WSDDatasetBuilder(cache_path, dictionary, use_synsets=use_synsets, keep_string_data=True, lang=lang)
        builder.add_raganato(
            xml_path=xml_path,
            max_length=max_length,
            input_keys=input_keys,
            tagset=tagset,
            on_error=on_error,
            quiet=quiet,
            read_by=read_by,
        )
        builder.finalize()
        inst = cls(cache_path, dictionary, target_classes='offsets', add_monosemous=add_monosemous, lazy=lazy)
        inst._from_tmp = from_tmp
        return inst

    def __init__(
            self,
            path: str,
            dictionary: Optional[Dictionary] = None,
            target_classes: str = "offsets",
            add_monosemous: bool = False,
            shuffle: bool = False,
            lazy: bool = False,
    ) -> None:

        self._loaded = False

        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as json_hdl:
                self.metadata = json.load(json_hdl)
        else:
            self.metadata = {"lang": "en"}

        self.use_synsets = target_classes == 'offsets'

        if not dictionary:
            dictionary = Dictionary.load(DEFAULT_DICTIONARY)
        self.dictionary = dictionary

        self.lemma_pos_dictionary = ResourceManager.get_lemma_pos_dictionary(self.lang)

        self.remap_senses = None
        if target_classes == 'offsets':
            self.output_dictionary = ResourceManager.get_offsets_dictionary()
            self.lemma_pos_to_possible_senses = ResourceManager.get_lemma_pos_to_possible_offsets(self.lang)
        elif target_classes == 'sensekeys':
            self.output_dictionary = ResourceManager.get_sensekeys_dictionary()
            self.lemma_pos_to_possible_senses = ResourceManager.get_lemma_pos_to_possible_sensekeys(self.lang)
        elif target_classes == 'bnids':
            self.output_dictionary = ResourceManager.get_offsets_dictionary()
            #self.remap_senses = ResourceManager.get_index_remap_offset_bnids()
            self.lemma_pos_to_possible_senses = ResourceManager.get_lemma_pos_to_possible_offsets(self.lang)
        else:
            raise ValueError('target_classes must be in {"sensekeys", "offsets", "bnids"} but was"' + target_classes +
                             '" instead')

        self.add_monosemous = add_monosemous
        self.shuffle = shuffle

        self.path = os.path.abspath(path)

        vectors_path = os.path.join(path, 'vectors.hdf5')
        gold_path = os.path.join(path, 'gold.pkl')
        oov_dictionary_path = os.path.join(path, 'oov.pkl')

        self._h5_file = h5py.File(vectors_path, mode="r")
        self._token_data_h5 = self._h5_file['.']['token_data']
        self._token_data_mem = None
        self._seq_data_h5 = self._h5_file['.']['seq_data']
        self._seq_data_mem = None
        with open(gold_path, 'rb') as pkl:
            self.gold = pickle.load(pkl)
        if os.path.exists(oov_dictionary_path):
            with open(oov_dictionary_path, 'rb') as pkl:
                self.oov_dictionary = pickle.load(pkl)
        else:
            self.oov_dictionary = None

        self.original_sizes = self.seq_data()[:, 1]
        self.target_sizes = None
        self._from_tmp = False

        if not lazy:
            self.load_in_memory()

        self.embeddings = None
        self.trg_manager = TargetManager('wsd')

    @property
    def lang(self):
        return self.metadata['lang']

    @property
    def sizes(self):
        if self.target_sizes is None:
            return self.original_sizes
        else:
            return self.target_sizes

    def token_data(self):
        if self._token_data_mem is None:
            return self._token_data_h5
        else:
            return self._token_data_mem

    def seq_data(self):
        if self._seq_data_mem is None:
            return self._seq_data_h5
        else:
            return self._seq_data_mem

    def load_in_memory(self):
        if self._loaded:
            pass
        else:
            self._token_data_mem = self._token_data_h5[()]
            self._seq_data_mem = self._seq_data_h5[()]
            self._loaded = True
        return self

    def clear_from_memory(self):
        self._token_data_mem = None
        self._seq_data_mem = None
        self._loaded = False
        return self

    def __del__(self):
        pass
        # self._h5_file.close()
        # if self._from_tmp:
        #     shutil.rmtree(self.path)

    def __len__(self):
        return len(self.sizes)

    def _get_item_data(self, item):
        seq_start, seq_len = self.seq_data()[item].tolist()
        data = self.token_data()[seq_start:seq_start+seq_len]
        return seq_start, seq_len, data

    def __getitem__(self, item):

        # for negative indexing support
        if item < 0:
            item += len(self)

        seq_start, seq_len, data = self._get_item_data(item)
        # seq_start, seq_len = self.seq_data()[item].tolist()
        # data = self.token_data()[seq_start:seq_start+seq_len]

        lemma_pos = data['lemma_pos'].astype(np.int64)
        all_ = [self.lemma_pos_to_possible_senses[int(l)] for l in lemma_pos]
        if self.remap_senses:
            all_ = [remove_dup([self.remap_senses[s] for s in ss]) for ss in all_]

        gold = []
        ids = []
        target = []
        for g_idx, a in zip(data['gold_indices'], all_):
            if g_idx >= 0:
                trg_idx, g = self.gold[g_idx]
            else:
                trg_idx = None
                if self.add_monosemous and len(a) == 1:
                    g = list(a)
                else:
                    g = [self.output_dictionary.unk()]

            if g:
                tgt = random.choice(g)
            else:
                tgt = self.output_dictionary.unk()

            if self.remap_senses:
                tgt = self.remap_senses[tgt]
                g = remove_dup([self.remap_senses[s] for s in g])
            target.append(tgt)
            gold.append(g)
            ids.append(trg_idx)

        target = torch.LongTensor(target)
        tokens_str = []
        for i, tkn in enumerate(data['token']):
            i += seq_start
            if self.oov_dictionary:
                s = self.oov_dictionary.get(i)
                if s is None:
                    s = self.dictionary.symbols[tkn]
            else:
                s = self.dictionary.symbols[tkn]
            tokens_str.append(s)

        sample = {
            'id' : item,
            'ntokens' : seq_len,
            'tokens' : torch.from_numpy(data['token'].astype(np.int64)),
            'tokens_str': tokens_str,
            'lemma_pos' : torch.from_numpy(lemma_pos),
            'senses' : {
                'target': target,
                'gold': gold,
                'ids': ids,
                'all': all_,
            },
            'lang': self.lang
        }

        if self.embeddings is not None:
            embedded = torch.from_numpy(self.embeddings[seq_start:seq_start+seq_len])
            sample['cached_vectors'] = embedded

        return sample

    def __iter__(self):
        # hacky solution
        for i in range(len(self)):
            yield self[i]

    def batch_lists(self, lists, max_len=None, pad_value=None):
        if not max_len:
            max_len = max(lists, key=len)
        samples = []
        for l in lists:
            pad = [pad_value] * (max_len - len(l))
            samples.append(l + pad)
        return samples

    def batch_tensors(self, tensors, max_len=None, pad_value=1):
        if not max_len:
            max_len = max(tensors, key=lambda t: t.size(0))
        samples = []
        for t in tensors:
            pad = torch.empty(max_len-t.size(0), dtype=torch.int64).fill_(pad_value)
            samples.append(torch.cat((t, pad), dim=0))
        batch = torch.stack(samples, dim=0)
        return batch

    def collater(self, samples:List[dict]) -> dict:
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `senses` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        max_len = max(s['ntokens'] for s in samples)

        src_lengths = torch.LongTensor([s["ntokens"] for s in samples])
        src_tokens = self.batch_tensors(
            [s['tokens'] for s in samples], max_len=max_len, pad_value=self.dictionary.pad())
        lemmas = self.batch_tensors(
            [s['lemma_pos'] for s in samples], max_len=max_len, pad_value=self.lemma_pos_dictionary.pad()
        )
        senses = self.batch_tensors(
            [s['senses']['target'] for s in samples], max_len=max_len, pad_value=self.output_dictionary.pad())

        all_senses = self.batch_lists([s['senses']['all'] for s in samples], max_len=max_len, pad_value={self.output_dictionary.pad()})
        target_ids = self.batch_lists([s['senses']['ids'] for s in samples], max_len=max_len, pad_value=None)
        src_tokens_str = self.batch_lists([s['tokens_str'] for s in samples], max_len=max_len, pad_value=self.dictionary.pad_word)
        gold = self.batch_lists([s['senses']['gold'] for s in samples], max_len=max_len, pad_value={self.output_dictionary.pad()})
        langs = [s['lang'] for s in samples]
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "ntokens": sum(s["ntokens"] for s in samples),
            'lemma_pos': lemmas,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "src_tokens_str": src_tokens_str,
                "langs": langs,
            },
            "senses": {
                "target": senses,
                "gold": gold,
                "all": all_senses,
                "ids": target_ids,
            }
        }

        if self.embeddings is not None and samples[0].get('cached_vectors') is not None:
            dtype = samples[0]['cached_vectors'].dtype
            vectors = []
            for s in samples:
                vec = s['cached_vectors']
                vec = torch.cat([vec, torch.zeros(max_len - vec.shape[0], *vec.shape[1:], dtype=dtype)])
                vectors.append(vec)
            batch['net_input']['cached_vectors'] = torch.stack(vectors, 0)

            batch = self.trg_manager.remove_non_targets(batch, self.output_dictionary.nspecial)

        return batch

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = max(num_tokens // tgt_len, 1)
        tokens = self.dictionary.dummy_sentence(tgt_len + 2)
        lemmas = self.lemma_pos_dictionary.dummy_sentence(tgt_len + 2)
        senses = self.output_dictionary.dummy_sentence(tgt_len + 2)

        batch = self.collater([{
            'id': i,
            'ntokens': tokens.size(0),
            'tokens': tokens,
            'tokens_str': [self.dictionary.unk_word] * tokens.size(0),
            'lemma_pos': lemmas,
            'lang': 'en',
            'senses': {
                'target': senses,
                'gold': [{self.output_dictionary.unk()}] * tokens.size(0),
                'ids': [None] * tokens.size(0),
                'all': [{self.output_dictionary.unk()}] * tokens.size(0),
            }
        } for i in range(bsz)])
        batch['dummy'] = True
        return batch

    def cache_embeddings(self, embeddings):
        self.embeddings = embeddings
        target_sizes = []
        for sample in self:
            target_tensor = sample['senses']['target']
            flat_mask = (target_tensor > self.output_dictionary.nspecial).view(-1)
            target_sizes.append(flat_mask.long().sum().item())
        self.target_sizes = target_sizes

    def clear_embeddins(self):
        self.embeddings = None

    def serialize_embeddings(self, name, overwrite=False):
        assert self.embeddings is not None
        embeddings_folder = os.path.join(self.path, 'embeddings')
        if not os.path.exists(embeddings_folder):
            os.mkdir(embeddings_folder)
        embeddings_file_path = os.path.join(embeddings_folder, name)
        if not overwrite and os.path.exists(embeddings_file_path):
            raise FileExistsError('{} already exists!'.format(embeddings_file_path))
        else:
            np.save(embeddings_file_path, self.embeddings)
        return embeddings_folder

    def load_embeddings(self, name):
        embeddings_file_path = os.path.join(self.path, 'embeddings', name + '.npy')
        self.cache_embeddings(np.load(embeddings_file_path))

    def has_cached_embeddings(self, name):
        embeddings_file_path = os.path.join(self.path, 'embeddings', name + '.npy')
        return os.path.exists(embeddings_file_path)

    def embed(
            self,
            embedder: BERTEmbedder,
            layers: Tuple[int] = (-1,),
            ntokens: int = 2000,
            sum_states: bool = True,
            progressbar: bool = False,
    ) -> torch.Tensor:

        embedder = embedder.eval()
        old_layers = embedder.layers
        embedder.layers = layers

        nvecs = self.token_data().shape[0]
        dim = embedder.embedding_dim

        embeddings = torch.zeros(nvecs, dim, 1 if sum_states else len(layers), dtype=torch.float32)
        ordered_indices = np.argsort(self.sizes)[::-1]
        seq_starts = self.seq_data()[:, 0]

        example_buffer_ids = []
        example_buffer_size = 0
        batch, batch_ids = None, None

        def process_batch():
            with torch.no_grad():
                states = embedder(batch['net_input']['src_tokens_str'])['inner_states']
            states = torch.stack(states, -1).cpu()
            if sum_states:
                states = states.sum(-1).unsqueeze(-1)

            for ord_ex_in_b, idx_ex_in_b in enumerate(batch_ids):

                seq_start = seq_starts[idx_ex_in_b]
                length = self.sizes[idx_ex_in_b]
                seq_end = seq_start + length
                vec = states[ord_ex_in_b, :length]
                embeddings[seq_start:seq_end] = vec

        it = zip(self.sizes, ordered_indices)
        if progressbar:
            it = tqdm(it, total=len(self))

        for len_ex, idx_ex in it:
            example_buffer_ids.append(idx_ex)
            example_buffer_size += len_ex
            if example_buffer_size < ntokens:
                batch, batch_ids = None, None
            elif example_buffer_size == ntokens:
                batch_ids = example_buffer_ids
                example_buffer_ids = []
                example_buffer_size = 0
                batch = self.collater([self[idx] for idx in batch_ids])
            else: # > ntokens
                batch_ids = example_buffer_ids[:-1]
                example_buffer_ids = example_buffer_ids[-1:]
                example_buffer_size = len_ex
                batch = self.collater([self[idx] for idx in batch_ids])
            if batch is not None:
                process_batch()

        if example_buffer_ids:
            batch_ids = example_buffer_ids
            batch = self.collater([self[idx] for idx in batch_ids])
            process_batch()

        embedder.layers = old_layers

        return embeddings.numpy()

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        return [self[i] for i in indices]

class WSDConcatDataset(FairseqDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, shuffle=False):
        super(WSDConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.sizes = np.concatenate([d.sizes for d in self.datasets], axis=0)

        self.get_dummy_batch = self.datasets[0].get_dummy_batch
        self.collater = self.datasets[0].collater
        self.shuffle = shuffle

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def prefetch(self, indices):
        return [self[index] for index in indices]

    def supports_prefetch(self):
        return all(d.supports_prefetch() for d in self.datasets)

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

_ud_to_wn = {
    "NOUN": "n",
    "VERB": "v",
    "ADV": "r",
    "ADJ": "a"
}
