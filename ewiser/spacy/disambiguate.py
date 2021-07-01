import math

from collections import namedtuple, deque
from typing import Optional

from nltk.corpus import wordnet as wn
import numpy as np
import torch

from ewiser.fairseq_ext.data.dictionaries import Dictionary, ResourceManager, DEFAULT_DICTIONARY
from ewiser.fairseq_ext.data.utils import make_offset
from ewiser.fairseq_ext.models.sequence_tagging import LinearTaggerModel
from ewiser.fairseq_ext.modules.logit_convolution import repack_sparse_tensor
from spacy.tokens.token import Token

_FakeTask = namedtuple('_FakeTask', ('dictionary', 'output_dictionary', 'kind'))

UD_WNPOS = {
    'NOUN': 'n',
    'VERB': 'v',
    'ADJ': 'a',
    'ADV': 'r'
}

def entropy_getter(token):

    n = len(token._.offsets_distribution)
    if n < 2:
        return 0.
    else:
        probs = np.array(list(token._.offsets_distribution.values()))
        entropy = - np.sum(probs * np.log(probs))
        normalized_entropy = entropy / np.log(n)
        return normalized_entropy

class DisambiguatorInternals:

    disambiguator: 'Disambiguator'
    logits: torch.FloatTensor
    logits_z: Optional[torch.FloatTensor]

    def __init__(self, disambiguator: 'Disambiguator', token: Token,  logits=None, logits_z=None):
        self.disambiguator = disambiguator
        self.token = token
        self.logits = logits
        self.logits_z = logits_z

    @property
    def dictionary(self):
        return self.disambiguator.output_dictionary

    def _logit_for_synset(self, logits, offset_or_synset):
        if isinstance(offset_or_synset, str):
            offset = offset_or_synset
        else:
            offset = make_offset(offset_or_synset)
        idx = self.dictionary.index(offset)
        return logits[idx].item()

    def _logits_topk(self, logits, k=10):
        scores, topk_indices = torch.topk(logits, k)
        offsets = [self.dictionary.symbols[idx.item()] for idx in topk_indices]
        synsets = [wn.synset_from_pos_and_offset(o[-1], int(o[3:-1])) for o in offsets]
        return synsets, scores

    def logit_for_synset(self, offset):
        return self._logit_for_synset(self.logits, offset)

    def logits_topk(self, k=10):
        return self._logits_topk(self.logits, k)

    def logit_z_for_synset(self, offset):
        if self.logits_z is not None:
            return self._logit_for_synset(self.logits_z, offset)
        else:
            return self._logit_for_synset(self.logits, offset)

    def logits_z_topk(self, k=10):
        if self.logits_z is not None:
            return self._logits_topk(self.logits_z, k)
        else:
            return self._logits_topk(self.logits, k)

    def adjacency_topk(self, k=10):
        offset = self.token._.offset
        assert self.disambiguator.model.decoder.structured_logits is not None
        if hasattr(self.disambiguator, '_A'):
            idx = self.dictionary.index(offset)
            row = self.disambiguator._A[idx].to_dense()
            row[:2] = -1e7
            sorted_weights, sorted_indices = torch.topk(row, k)
            mask = sorted_weights != 0.
            sorted_weights, sorted_indices = sorted_weights[mask], sorted_indices[mask]
            self_w = 1 if self.disambiguator.model.decoder.structured_logits.self_loops is None \
                else self.disambiguator.model.decoder.structured_logits.self_loops[idx]
            sorted_offsets = [self.dictionary.symbols[i.item()] for i in sorted_indices]
            sorted_synsets = [wn.synset_from_pos_and_offset(o[-1], int(o[3:-1])) for o in sorted_offsets]
            return (sorted_synsets, sorted_weights), (wn.synset_from_pos_and_offset(offset[-1], int(offset[3:-1])), self_w)

        else:
            params = self.disambiguator.model.decoder.structured_logits.adjacency_pars
            A = repack_sparse_tensor(*params).detach().cpu()
            self.disambiguator._A = A
            return self.adjacency_topk(k)

class Disambiguator(torch.nn.Module):

    _wn = None
    _k = 10

    def __init__(self, checkpoint, lang='en', dictionary=None, save_wsd_details=True, maxlen=100, batch_size=5):
        super().__init__()

        try:
            self._set_spacy_extensions()
        except ValueError:
            pass

        self.lang = lang
        if lang != 'en':
            self.lemma_pos_dictionary = ResourceManager.get_lemma_pos_dictionary(lang=lang)
            self.lemma_pos_to_possible_offsets = ResourceManager.get_lemma_pos_to_possible_offsets(lang=lang)

        self.dictionary = dictionary or Dictionary.load(DEFAULT_DICTIONARY)
        self.output_dictionary = ResourceManager.get_offsets_dictionary()
        self.model = self._load_model(
            checkpoint,
            _FakeTask(self.dictionary, self.output_dictionary, 'wsd')
        )
        self.save_wsd_details = save_wsd_details
        self.maxlen = maxlen
        self.batch_size = batch_size

    def forward(self, doc):
        next(self.pipe([doc], batch_size=self.batch_size))

        return doc

    def pipe(self, docs, batch_size=5):
        tokens_indices = []
        tokens_str = []
        tokens_spacy = []

        n_added_to_deque = 0
        n_processed = 0
        docs_to_yield = deque()

        for doc in docs:

            data_doc = self._make_input_from_doc(doc)

            n_added_to_deque += len(data_doc[0])
            docs_to_yield.append((doc, n_added_to_deque))

            tokens_indices.extend(data_doc[0])
            tokens_str.extend(data_doc[1])
            tokens_spacy.extend(data_doc[2])

            # process if we reached the required batch dimension
            while len(tokens_spacy) >= batch_size:
                batch_indices, tokens_indices = tokens_indices[:batch_size], tokens_indices[batch_size:]
                batch_str, tokens_str = tokens_str[:batch_size], tokens_str[batch_size:]
                batch_spacy, tokens_spacy = tokens_spacy[:batch_size], tokens_spacy[batch_size:]

                batch_maxlen = max(len(x) for x in batch_indices)
                batch_indices = \
                    [indices + [self.dictionary.pad_index] * (batch_maxlen - len(indices)) for indices in
                     batch_indices]
                batch_indices = torch.LongTensor(batch_indices).to(self.device)

                with torch.no_grad():
                    logits, additional_data = self.model(src_tokens=batch_indices, src_tokens_str=batch_str)
                self._tag_docs_from_model_output(logits, batch_spacy, additional_data=additional_data)

                n_processed += len(batch_spacy)

            # yield docs already processed entirely
            while docs_to_yield and docs_to_yield[0][1] <= n_processed:
                yield docs_to_yield.popleft()[0]

        while tokens_spacy:
            batch_indices, tokens_indices = tokens_indices[:batch_size], tokens_indices[batch_size:]
            batch_str, tokens_str = tokens_str[:batch_size], tokens_str[batch_size:]
            batch_spacy, tokens_spacy = tokens_spacy[:batch_size], tokens_spacy[batch_size:]

            batch_maxlen = max(len(x) for x in batch_indices)
            batch_indices = \
                [indices + [self.dictionary.pad_index] * (batch_maxlen - len(indices)) for indices in batch_indices]
            batch_indices = torch.LongTensor(batch_indices).to(self.device)

            with torch.no_grad():
                logits, additional_data = self.model(src_tokens=batch_indices, src_tokens_str=batch_str)
            self._tag_docs_from_model_output(logits, batch_spacy, additional_data=additional_data)

            n_processed += len(batch_spacy)

        # yield docs already processed entirely
        while docs_to_yield:
            yield docs_to_yield.popleft()[0]

    def _make_input_from_doc(self, doc):

        tokens_str = [t.text for t in doc]
        tokens_indices = [self.dictionary.index(t) for t in tokens_str]
        tokens_spacy = [t for t in doc]

        nseq = math.ceil(len(tokens_str) / self.maxlen)

        sequences_str = np.array_split(tokens_str, nseq)
        sequences_indices = np.array_split(tokens_indices, nseq)
        sequences_spacy = np.array_split(tokens_spacy, nseq)

        sequences_str = [seq.tolist() for seq in sequences_str]
        sequences_indices = [seq.tolist() for seq in sequences_indices]
        sequences_spacy = [seq.tolist() for seq in sequences_spacy]

        return sequences_indices, sequences_str, sequences_spacy

    def _tag_docs_from_model_output(self, logits, sequences_spacy, additional_data=None):

        logits = logits.detach().cpu()
        logits[:,:,0:2] = -1e7
        if (additional_data is not None) and (additional_data.get('prelogits') is not None):
            prelogits = additional_data['prelogits'].detach().cpu()
            prelogits[:,:,0:2] = -1e7
        else:
            prelogits = None


        for i, seq in enumerate(sequences_spacy):
            for j, t in enumerate(seq):

                logits_token = logits[i, j]

                lemma = t._.lemma_preset_else_spacy.lower()
                pos = t._.pos_preset_else_spacy
                wnpos = UD_WNPOS.get(pos)

                if lemma and wnpos:
                    if self.lang == 'en':
                        synsets = wn.synsets(lemma, wnpos)
                        offsets = [make_offset(s) for s in synsets]
                    else:
                        lemma_pos = lemma + '#' + wnpos
                        lemma_pos_index = self.lemma_pos_dictionary.index(lemma_pos)
                        offsets_indices = self.lemma_pos_to_possible_offsets[lemma_pos_index]
                        offsets = [self.output_dictionary.symbols[i] for i in offsets_indices]
                        offsets = [o for o in offsets if o.startswith('wn:')]

                    if not offsets:
                        continue
                    indices = np.array([self.output_dictionary.index(o) for o in offsets])
                else:
                    continue

                logits_synsets = logits_token[indices]
                index = torch.max(logits_synsets, -1).indices.item()
                t._.offset = offsets[index]

                if self.save_wsd_details:
                    internals = t._.disambiguator_internals = DisambiguatorInternals(self, t)
                    internals.logits = logits_token
                    if prelogits is not None:
                        internals.logits_z = prelogits[i, j]

    def _set_spacy_extensions(self):

        def synset_getter(token):

            if not Disambiguator._wn:
                from nltk.corpus import wordnet as wn
                Disambiguator._wn = wn

            else:
                wn = Disambiguator._wn

            offset = token._.offset
            if offset:
                return wn.synset_from_pos_and_offset(offset[-1], int(offset[3:-1]))
            else:
                return None


        from spacy.tokens import Doc, Token
        Doc.set_extension('lang', default='en')

        Token.set_extension('lemma_preset_', default=None)
        Token.set_extension('pos_preset_', default=None)

        Token.set_extension('lemma_preset_else_spacy', getter=lambda t: t._.lemma_preset_ or t.lemma_)
        Token.set_extension('pos_preset_else_spacy', getter=lambda t: t._.pos_preset_ or t.pos_)

        Token.set_extension('offset', default=None)
        Token.set_extension('synset', getter=synset_getter)
        Token.set_extension('disambiguator_internals', default=None)

    def _load_model(self, checkpoint, task):
        data = torch.load(checkpoint, map_location='cpu')
        args = data['args']
        if args.arch == 'linear_seq':
            model = LinearTaggerModel.build_model(data['args'], task).eval()
        else:
            raise ValueError
        model.load_state_dict(data['model'])
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    def enable(self, nlp, name):
        import spacy
        version = int(spacy.__version__.split('.')[0])
        if version < 3:
            nlp.add_pipe(self, last=True)
        else:
            from spacy.language import Language
            @Language.factory(name)
            def wsd(nlp, name):
                return self
            nlp.add_pipe(name, last=True)


if __name__ == '__main__':

    from argparse import ArgumentParser
    from spacy import load

    parser = ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--lang', default='en', choices='en de fr it es'.split())
    parser.add_argument('--spacy', default='', type=str)
    args = parser.parse_args()

    wsd = Disambiguator(args.checkpoint, lang=args.lang)
    nlp = load(args.spacy or args.lang, disable=['parser', 'ner'])
    wsd.enable(nlp, "wsd")

    print('Input a sentence then press Enter:')
    while True:
        line = input()
        doc = nlp(line)
        for w in doc:
            if w._.offset:
                print(w.text, w.lemma_, w.pos_, w._.offset, w._.synset.definition())
        print()




