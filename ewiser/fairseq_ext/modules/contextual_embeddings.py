import abc
import os
from typing import Union, List

import torch

class FakeInput(torch.nn.Module):

    def __init__(self, embedding_dim, padding_idx=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx


class BaseContextualEmbedder(torch.nn.Module, metaclass=abc.ABCMeta):

    embedding_dim: int
    n_hidden_states: int
    retrain_model: bool

    def __init__(self, retrain_model: bool = False):
        super().__init__()
        self.retrain_model = retrain_model

    def forward(
            self,
            src_tokens: Union[None, torch.LongTensor] = None,
            src_tokens_str: Union[None, List[List[str]]] = None,
            batch_major: bool = True,
            **kwargs
    ):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    @property
    def embed_dim(self):
        return self.embedding_dim

    @property
    def embedded_dim(self):
        return self.embedding_dim

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)

class BERTEmbedder(BaseContextualEmbedder):

    DEFAULT_MODEL = 'bert-base-cased'

    @staticmethod
    def _do_imports():
        import pytorch_pretrained_bert as bert
        return bert

    def __init__(
        self,
        name: Union[str, None] = None,
        weights: str = "",
        retrain_model=False,
        layers=(-1,)
    ):
        assert not retrain_model
        super(BERTEmbedder, self).__init__(retrain_model=False)

        assert not retrain_model

        if not name:
            name = self.DEFAULT_MODEL

        self.name = name
        self.retokenize = True
        self.realign = 'MEAN'
        self.uncased = 'uncased' in name
        self.layers = layers

        bert = self._do_imports()
        self.bert_tokenizer = bert.BertTokenizer.from_pretrained(name, do_lower_case=self.uncased)
        self.bert_model = bert.BertModel.from_pretrained(name)

        if weights:

            state = torch.load(weights)['state']
            state = {".".join(k.split('.')[1:]): v for k, v in state.items() if k.startswith('bert.')}
            self.bert_model.load_state_dict(state)
            self.name = self.name + "-" + os.path.split(weights)[-1]

        for par in self.parameters():
            par.requires_grad = False

    def _subtokenize_sequence(self, tokens):
        split_tokens = []
        merge_to_previous = []
        for token in tokens:
            for i, sub_token in enumerate(self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)):
                split_tokens.append(sub_token)
                merge_to_previous.append(i > 0)

        split_tokens = ['[CLS]'] + split_tokens + ['[SEP]']
        return split_tokens, merge_to_previous

    def _convert_to_indices(self, subtokens, maxlen=-1):
        unpadded_left = [self.bert_tokenizer.vocab[st] for st in subtokens]
        if maxlen > 0:
            unpadded_left = unpadded_left[:maxlen]
            padded_left = unpadded_left + \
                [0] * (maxlen - len(unpadded_left))
        return torch.LongTensor(padded_left)

    def forward(self, src_tokens_str: List[List[str]], src_tokens=None, batch_major=True, padding_str='<pad>',
                **kwargs):

        if self.retokenize:

            src_tokens_str = [[t for t in seq if t != '<pad>'] for seq in src_tokens_str]
            subtoken_str = []
            merge_to_previous = []
            for seq in src_tokens_str:
                ss, mm = self._subtokenize_sequence(seq)
                subtoken_str.append(ss)
                merge_to_previous.append(mm)

            max_token_len = max(map(len, src_tokens_str))
            subtoken_len = list(map(len, subtoken_str))
            max_subtoken_len = max(subtoken_len)

            input_ids = torch.stack([self._convert_to_indices(seq, max_subtoken_len) for seq in subtoken_str], dim=0).to(self.device)

            attention_mask = input_ids > 0
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)
            for i, l in enumerate(subtoken_len):
                token_type_ids[i, l - 1:] = 1

            with torch.set_grad_enabled(self.retrain_model and not self.training):

                hidden_states, *_ = self.bert_model.eval().forward(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    output_all_encoded_layers=True
                )
                hidden_states = [hidden_states[l] for l in self.layers]
                hidden_states = torch.cat(hidden_states, -1)
                hidden_states = hidden_states[:, 1:]

            if self.realign == 'MEAN':

                contextual_embeddings = torch.zeros(
                    len(src_tokens_str),
                    max_token_len,
                    hidden_states.shape[-1],
                    dtype=torch.float32,
                    device=self.device
                )

                for n_seq, merge_seq in enumerate(merge_to_previous):

                    n_token = -1
                    n_subtoken_per_token = 1
                    for n_subtoken, merge in enumerate(merge_seq):

                        if merge:
                            n_subtoken_per_token += 1
                        else:
                            n_subtoken_per_token = 1
                            n_token += 1

                        prev = contextual_embeddings[n_seq, n_token]
                        next = hidden_states[n_seq, n_subtoken]
                        contextual_embeddings[n_seq, n_token] = prev + (next - prev) / n_subtoken_per_token

            elif self.realign == 'MEANv2':

                import torch_scatter

                scatter_index = [merge_seq + [-1] * (max_subtoken_len - len(merge_seq) - 1) for merge_seq in
                                 merge_to_previous]
                scatter_index = torch.LongTensor(scatter_index, device=self.device)
                mask_flat = (scatter_index != -1).view(-1)
                scatter_index = (scatter_index == 0).long()
                scatter_index[:, 0] = torch.arange(0, len(merge_to_previous) * max_token_len, max_token_len, device=self.device)
                scatter_index = torch.cumsum(scatter_index, dim=-1)
                scatter_index_flat = scatter_index.view(-1)[mask_flat]

                raw_output_flat = raw_output.contiguous().view(-1, raw_output.size(-1))[mask_flat]
                contextual_embeddings = torch_scatter.scatter_mean(
                    raw_output_flat,
                    scatter_index_flat,
                    -2,
                    dim_size=raw_output.size(0) * max_token_len
                )
                contextual_embeddings = contextual_embeddings.view(raw_output.size(0), max_token_len, -1)

            elif self.realign == 'FIRST':
                contextual_embeddings = torch.zeros(len(src_tokens_str), max_token_len, self.embedding_dim,
                                                    dtype=torch.float32).to(self.device)
                for n_seq, merge_seq in enumerate(merge_to_previous):

                    ids = [i for i, merge in enumerate(merge_seq) if not merge]

                    for n_token, n_subtoken in enumerate(ids):

                        contextual_embeddings[n_seq, n_token] = hidden_states[n_seq, n_subtoken]

            else:
                raise

        if not batch_major:
            contextual_embeddings = contextual_embeddings.transpose(0, 1)

        contextual_embeddings = torch.chunk(contextual_embeddings, len(self.layers), -1)
        return {'inner_states': contextual_embeddings}

    @property
    def embedding_dim(self):
        return self.bert_model.encoder.layer[-1].output.dense.out_features

    @property
    def n_hidden_states(self):
        return 1

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda


class TransformerEmbedder(BaseContextualEmbedder):

    DEFAULT_MODEL = 'bert-base-cased'

    @staticmethod
    def _do_imports():
        import transformers
        return transformers

    def __init__(
        self,
        name: Union[str, None] = None,
        retrain_model=False,
        layers=(-1,)
    ):
        assert not retrain_model
        super(TransformerEmbedder, self).__init__(retrain_model=False)

        assert not retrain_model

        if not name:
            name = self.DEFAULT_MODEL

        self.name = name
        self.retokenize = True
        self.uncased = 'uncased' in name
        self.layers = layers

        transformers = self._do_imports()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        config = transformers.AutoConfig.from_pretrained(name)
        config.output_hidden_states = True
        self.model = transformers.AutoModel.from_pretrained(name, config=config)

        self.realign = 'MEAN'

        fake = 123456789
        with_special = self.tokenizer.build_inputs_with_special_tokens([fake])
        fake_idx = with_special.index(fake)
        pre, post = with_special[:fake_idx], with_special[fake_idx+1:]
        self.pre = self.tokenizer.convert_ids_to_tokens(pre)
        self.post = self.tokenizer.convert_ids_to_tokens(post)

        self.embedding_dim = self.model.config.hidden_size

        for par in self.parameters():
            par.requires_grad = retrain_model

    def _subtokenize_sequence(self, tokens):
        split_tokens = []
        merge_to_previous = []
        for token in tokens:
            for i, sub_token in enumerate(self.tokenizer.tokenize(token, add_prefix_space=True)):
                split_tokens.append(sub_token)
                merge_to_previous.append(i > 0)

        split_tokens = self.pre + split_tokens + self.post
        return split_tokens, merge_to_previous

    def _convert_to_indices(self, subtokens, maxlen=-1):
        unpadded_left = self.tokenizer.convert_tokens_to_ids(subtokens)
        if maxlen > 0:
            unpadded_left = unpadded_left[:maxlen]
            padded_left = unpadded_left + \
                [self.tokenizer.pad_token_id] * (maxlen - len(unpadded_left))
        return torch.LongTensor(padded_left)

    def forward(self, src_tokens_str: List[List[str]], src_tokens=None, batch_major=True, padding_str='<pad>',
                **kwargs):

        if self.retokenize:

            src_tokens_str = [[t for t in seq if t != '<pad>'] for seq in src_tokens_str]
            subtoken_str = []
            merge_to_previous = []
            for seq in src_tokens_str:
                ss, mm = self._subtokenize_sequence(seq)
                subtoken_str.append(ss)
                merge_to_previous.append(mm)


            max_token_len = max(map(len, src_tokens_str))
            subtoken_len = list(map(len, subtoken_str))
            max_subtoken_len = max(subtoken_len)

            input_ids = torch.stack([self._convert_to_indices(seq, max_subtoken_len) for seq in subtoken_str], dim=0).to(self.device)

            with torch.set_grad_enabled(self.retrain_model and not self.training):

                _, _, hidden_states, *_ = self.model.eval().forward(
                    input_ids=input_ids,
                )
                hidden_states = [hidden_states[l] for l in self.layers]
                hidden_states = torch.cat(hidden_states, -1)
                hidden_states = hidden_states[:, len(self.pre):]

            contextual_embeddings = torch.zeros(
                len(src_tokens_str),
                max_token_len,
                hidden_states.shape[-1],
                dtype=torch.float32,
                device=self.device
            )

            for n_seq, merge_seq in enumerate(merge_to_previous):

                n_token = -1
                n_subtoken_per_token = 1
                for n_subtoken, merge in enumerate(merge_seq):

                    if merge:
                        n_subtoken_per_token += 1
                    else:
                        n_subtoken_per_token = 1
                        n_token += 1

                    prev = contextual_embeddings[n_seq, n_token]
                    next = hidden_states[n_seq, n_subtoken]
                    contextual_embeddings[n_seq, n_token] = prev + (next - prev) / n_subtoken_per_token

        if not batch_major:
            contextual_embeddings = contextual_embeddings.transpose(0, 1)

        contextual_embeddings = torch.chunk(contextual_embeddings, len(self.layers), -1)
        return {'inner_states': contextual_embeddings}


    @property
    def n_hidden_states(self):
        return 1

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
