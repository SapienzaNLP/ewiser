import collections
import itertools
from typing import List, Union

import torch
from torch.nn import functional as F

from fairseq import options
from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, FairseqDecoder
from fairseq.models.transformer import Embedding
from fairseq.modules import CharacterTokenEmbedder, AdaptiveInput

from ewiser.fairseq_ext.modules.contextual_embeddings import BERTEmbedder, BaseContextualEmbedder, FakeInput, \
    TransformerEmbedder
from ewiser.fairseq_ext.modules.logit_convolution import StructuredLogits
from ewiser.fairseq_ext.modules.activations import penalized_tanh, swish, mish


def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
    from fairseq import utils

    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
    embed_dict = utils.parse_embedding(embed_path)
    utils.print_embed_overlap(embed_dict, dictionary)
    return utils.load_embedding(embed_dict, dictionary, embed_tokens)


class FairseqTaggerDecoder(FairseqDecoder):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__(dictionary)
        self.dictionary = dictionary

    def forward(self, tokens, precomputed_embedded=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        raise NotImplementedError

    def get_logits(self, net_output, log_probs, sample):
        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            out = self.adaptive_softmax
            out = self.adaptive_softmax.get_log_prob(net_output[0], sample['target'])
            return out.exp_() if not log_probs else out

        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            out = self.adaptive_softmax.get_log_prob(net_output[0], sample['target'])
            return out.exp_() if not log_probs else out

        logits = net_output[0].float()

        if getattr(self, 'sigsoftmax', True):
            sigmoid_logits = logits.sigmoid().log()
            logits = logits + sigmoid_logits

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq_ext."""
        return state_dict

class TaggerModel(BaseFairseqModel):
    """Base class for sequence labeling models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, embedder, decoder, kind, use_all_hidden=False, use_cached_vectors=False):
        super().__init__()
        self.embedder = embedder
        self.decoder = decoder
        self.kind = kind

        self.use_all_hidden = use_all_hidden
        self.use_cached_vectors = use_cached_vectors

        if embedder is not None:
            self.n_hidden_states = (embedder.n_hidden_states if use_all_hidden else 1)
            if self.n_hidden_states > 1:
                self.layer_weights = torch.nn.Parameter(torch.zeros(self.n_hidden_states))
                self.layer_weights.requires_grad = True
            else:
                self.layer_weights = None
        else:
            self.n_hidden_states = None
            self.layer_weights = None

        assert isinstance(self.decoder, FairseqDecoder)

    @staticmethod
    def add_args(parser):

        # Contextual embeddings args
        parser.add_argument("--context-embeddings", action="store_true",
            help="The hidden states of a pretrained model as contextual embeddings.")
        parser.add_argument("--context-embeddings-type", type=str, choices=['bert', 'transformers'], default='bert',
            help='The embedder to use. "bert" to use BertModel from pytorch_pretrained_bert, "transformers" to use AutoModel from transformers.')
        parser.add_argument('--context-embeddings-bert-model', type=str, default=BERTEmbedder.DEFAULT_MODEL,
            help='The transformers model identifier (e.g. "bert-base-cased").')
        parser.add_argument("--context-embeddings-layers", type=int, nargs='+', default=[-4,-3,-2,-1],
            help="Use the sum of the specified pretrained model's hidden layers.", )
        parser.add_argument("--context-embeddings-cache", action="store_true",
            help='Cache contextual embeddings.')

        # Tagger hidden dim
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
            help='Hidden dimension.')

        # Structured logits args
        parser.add_argument('--decoder-use-structured-logits', action='store_true',
            help='Enable EWISER\'s structured logits.')
        parser.add_argument('--decoder-structured-logits-edgelists', nargs='*', type=str,
            help="""
            Build the EWISER's A adjacency matrix from the following edgelist files.
            One edge per row, nodes are WordNet offsets. Weight can be specified.
            """.strip())
        parser.add_argument('--decoder-structured-logits-trainable', action='store_true',
            help='Refine the weights of A during training.')
        parser.add_argument('--decoder-structured-logits-renormalize', action='store_true',
            help='Always renormalize the weights of incoming edges to sum to 1.')

        # O initialization args 
        parser.add_argument('--decoder-output-pretrained', default='', type=str,
            help='Initialize the weights of the output matrix from the specified plaintext embeddings. No header.')
        parser.add_argument('--decoder-output-fixed', action='store_true',
            help='Freeze the output embeddings.')
        parser.add_argument('--decoder-output-pretrained-normalization', default='none', choices=['none', 'l2', 'scale', 'mean'],
            help='If reading the embeddings from file, renormalize them.')

        # Checkpoint restoring
        parser.add_argument('--only-load-weights', action='store_true',
            help="""
            If restoring a checkpoint with --restore_file, only load the weights and ignore args stored within the checkpoint.
            Useful if changing options, chiefly for unfreezing the output embeddings.
            """.strip())

        # Input matrix (not relevant for EWISER nor baseline)
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
            help='Input dimension (vector dim of the input embedding matrix). Ignored with --context-embeddings.')
        parser.add_argument('--decoder-embed-pretrained', type=str, metavar='P', default='',
            help='Use provided embeddings to initialize the embedding matrix. Ignored with --context-embeddings.')
        parser.add_argument('--decoder-embed-pretrained-freeze', action='store_true',
            help='Freeze the input embedding matrix. Ignored with --context-embeddings.')

        # Characted embedder args (not relevant for EWISER nor baseline)
        parser.add_argument('--character-embeddings', default=False, action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', type=int, metavar='N', default=4,
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', type=int, metavar='N', default=2,
                            help='number of highway layers for character token embeddder')

        # Adaptive input args (not relevant for EWISER nor baseline)
        parser.add_argument('--adaptive-input', default=False, action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')

    @classmethod
    def build_model_input(cls, args, dictionary):
        # make sure all arguments are present in older fairseq_ext

        args.context_embeddings = getattr(args, 'context_embeddings', False)
        args.context_embeddings_layers = getattr(args, 'context_embeddings_layers', [-1])

        args.max_source_positions = args.tokens_per_sample
        args.max_target_positions = args.tokens_per_sample

        if args.context_embeddings:
            if args.context_embeddings_type == 'bert':
                embed_tokens = BERTEmbedder(
                    args.context_embeddings_bert_model,
                    layers=args.context_embeddings_layers
                )

            elif args.context_embeddings_type == 'transformers':
                embed_tokens = TransformerEmbedder(
                    args.context_embeddings_bert_model,
                    layers=args.context_embeddings_layers
                )

            else:
                raise NotImplementedError
        elif args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(dictionary, eval(args.character_filters),
                                                  args.character_embedding_dim,
                                                  args.decoder_embed_dim,
                                                  args.char_embedder_highway_layers,
                                                  )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(len(dictionary), dictionary.pad(), args.decoder_input_dim,
                                         args.adaptive_input_factor, args.decoder_embed_dim,
                                         options.eval_str_list(args.adaptive_input_cutoff, type=int))
        else:
            args.decoder_embed_pretrained = getattr(args, 'decoder_embed_pretrained', '')
            if args.decoder_embed_pretrained:
                embed_tokens = load_pretrained_embedding_from_file(args.decoder_embed_pretrained, dictionary, args.decoder_input_dim)
            else:
                embed_tokens = Embedding(len(dictionary), args.decoder_input_dim, dictionary.pad())

        return embed_tokens

    @classmethod
    def build_model_decoder(cls, args, dictionary, output_dictionary, embed_tokens):
        raise NotImplementedError

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample[self.kind]["target"]

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        embedder = cls.build_model_input(args, task.dictionary)
        layers = getattr(args, 'context_embeddings_type', [-1])
        if isinstance(embedder, BaseContextualEmbedder):
            decoder = cls.build_model_decoder(args, task.dictionary, task.output_dictionary, FakeInput(embedder.embedding_dim))
            inst = cls(
                embedder,
                decoder,
                task.kind,
                use_all_hidden=len(layers) > 1,
                use_cached_vectors=True,
            )
        else:
            decoder = cls.build_model_decoder(args, task.dictionary, task.output_dictionary, embedder)
            inst = cls(
                None,
                decoder,
                task.kind,
                use_all_hidden=len(layers) > 1,
            )
        return inst

    def forward_encode_words(
            self,
            src_tokens=None,
            src_lengths=None,
            src_tokens_str=None,
            langs: Union[str, List[str]] = 'en'
    ):

        if self.embedder is None:
            return self.decoder(src_tokens=src_tokens, src_lengths=src_lengths, src_tokens_str=src_tokens_str)
        else:
            out = self.embedder(src_tokens=src_tokens, src_lengths=src_lengths, src_tokens_str=src_tokens_str)
        if self.use_all_hidden:
            states = out["inner_states"]
        else:
            states = out["inner_states"][-1:]
        if len(states) > 1:
            stacked = torch.stack(states, dim=0)
            precomputed_embedded = stacked.sum(0)
        else:
            precomputed_embedded = states[0]
        return precomputed_embedded

    def forward_encode_sequence(self, src_tokens=None, precomputed_embedded=None):
        return precomputed_embedded

    def forward_encode_head(self, src_tokens, precomputed_embedded=None):
        return self.decoder(src_tokens, precomputed_embedded=precomputed_embedded)

    def forward(
            self,
            src_tokens=None,
            src_lengths=None,
            src_tokens_str=None,
            langs: Union[str, List[str]] = 'en',
            cached_vectors: torch.Tensor = None,
    ):
        if self.use_cached_vectors and cached_vectors is not None:
            out = cached_vectors.sum(-1).to(src_tokens.device)
        else:
            out = self.forward_encode_words(src_tokens, src_lengths, src_tokens_str, langs)
        out = self.forward_encode_sequence(src_tokens, out)
        out = self.forward_encode_head(src_tokens, out)
        return out

    def forward_legacy(self, src_tokens=None, src_lengths=None, src_tokens_str=None, langs: Union[str, List[str]] = 'en'):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the decoder's output, typically of shape `(batch, seq_len, vocab)`
        """

        if self.embedder is None:
            return self.decoder(src_tokens=src_tokens, src_lengths=src_lengths, src_tokens_str=src_tokens_str)
        else:
            out = self.embedder(src_tokens=src_tokens, src_lengths=src_lengths, src_tokens_str=src_tokens_str)
            if self.use_all_hidden:
                states = out["inner_states"]
            else:
                states = out["inner_states"][-1:]
            if len(states) > 1:
                stacked = torch.stack(states, dim=0)
                weights = F.softmax(self.layer_weights, 0).view(-1, 1, 1, 1)
                precomputed_embedded = (weights * stacked).sum(0)
            else:
                precomputed_embedded = states[0]
        return self.decoder(src_tokens, precomputed_embedded=precomputed_embedded)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    def remove_head(self):
        """Removes the head of the model (e.g. the softmax layer) to conserve space when it is not needed"""
        raise NotImplementedError()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if (self.embedder is not None) and (isinstance(self.embedder, BaseContextualEmbedder)):
            for name in list(destination.keys()):
                if name.startswith('embedder'):
                    del destination[name]

        return destination

    def upgrade_state_dict(self, state, prefix=''):
        if (self.embedder is not None) and (isinstance(self.embedder, BaseContextualEmbedder)):
            self.embedder.state_dict(state, prefix + 'embedder.')

def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.)
    return m

class LinearDecoder(FairseqTaggerDecoder):

    def __init__(
            self,
            args, dictionary, embed_tokens, adjacency=None,):
        super().__init__(dictionary)

        args.decoder_output_dim = \
            getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
        args.decoder_norm = \
            getattr(args, 'decoder_norm', False)
        args.decoder_activation = \
            getattr(args, 'decoder_activation', 'relu')
        args.decoder_last_activation = \
            getattr(args, 'decoder_last_activation', True)
        args.decoder_output_pretrained = \
            getattr(args, 'decoder_output_pretrained', '')
        args.decoder_output_pretrained_normalization = \
            getattr(args, 'decoder_output_pretrained_normalization', 'none').lower()
        args.decoder_output_use_bias = \
            getattr(args, 'decoder_output_use_bias', True)

        if hasattr(embed_tokens, "embed_dim"):
            self.input_dim = embed_tokens.embedding_dim
        else:
            self.input_dim = args.decoder_embed_dim

        self.embed_tokens = embed_tokens

        assert args.decoder_layers >= 1
        self.linears = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(args.dropout)
        for i in range(args.decoder_layers - 1):
            in_features = self.input_dim if i == 0 else args.decoder_embed_dim
            out_features = args.decoder_embed_dim if i < (args.decoder_layers - 2) else args.decoder_output_dim
            self.linears.append(Linear(in_features, out_features))
        self.logits = torch.nn.Linear(
            self.input_dim if args.decoder_layers == 1 else args.decoder_output_dim,
            len(dictionary),
            bias=args.decoder_output_use_bias,
        )

        if args.decoder_output_pretrained:
            weight = load_pretrained_embedding_from_file(
                args.decoder_output_pretrained,
                dictionary,
                embed_dim=self.logits.weight.shape[1]
            ).weight
            if args.decoder_output_pretrained_normalization == 'none':
                ...
            elif args.decoder_output_pretrained_normalization == 'l2':
                data = weight.data
                norm = (data ** 2).sum(-1).view(data.size(0), 1) ** 0.5
                data /= norm
                weight.data = data
            elif args.decoder_output_pretrained_normalization == 'scale':
                data = weight.data
                mean = data.mean(0).view(1, data.size(1))
                data /= mean
                weight.data = data
            elif args.decoder_output_pretrained_normalization == 'mean':
                data = weight.data
                mean = data.mean(0).view(1, data.size(1))
                std = data.std(0).view(1, data.size(1))
                data -= mean
                data /= std
                weight.data = data
            else:
                raise
            self.logits.weight = weight
            args.decoder_output_pretrained = ''

        if getattr(args, 'decoder_output_fixed', False):
            self.logits.weight.requires_grad = False
            if hasattr(self.logits, 'bias') and (self.logits.bias is not None):
                self.logits.bias.requires_grad = False

        if adjacency is not None:

            self.structured_logits = StructuredLogits(
                adjacency,
                adjacency_trainable=getattr(args, 'decoder_structured_logits_trainable', False),
                renormalize=getattr(args, 'decoder_structured_logits_renormalize', False))
        else:
            self.structured_logits = None

        if args.decoder_norm:
            self.norm = torch.nn.BatchNorm1d(self.input_dim)
        else:
            self.norm = None

        if   args.decoder_activation == 'relu':
            self.activation = F.relu
        elif args.decoder_activation == 'swish':
            self.activation = swish
        elif args.decoder_activation == 'tanh':
            self.activation = torch.tanh
        elif args.decoder_activation == 'penalized_tanh':
            self.activation = penalized_tanh
        elif args.decoder_activation == 'mish':
            self.activation = mish
        else:
            raise NotImplementedError

        self.last_activation = args.decoder_last_activation

    def embedding_forward(self, tokens):
        out = self.embed_tokens(tokens)
        return out

    def encode_forward(self, embedded):
        out = embedded
        for i, lin in enumerate(self.linears, start=1):
            out = lin(out)
            out = self.dropout(out)
            if (i < len(self.linears)) or self.last_activation:
                out = self.activation(out)
        return out

    def head_forward(self, encoded):
        out = encoded
        out = self.logits(out)
        if self.structured_logits is not None:
            return self.structured_logits(out), out
        else:
            return out, out

    def forward(self, tokens, precomputed_embedded=None):
        if precomputed_embedded is None:
            out = self.embedding_forward(tokens)
        else:
            out = precomputed_embedded
        if self.norm is not None:
            size = out.size()
            out = self.norm(out.view(-1, out.size(-1)))
            out = out.view(*size)
        out = self.encode_forward(out)
        out = self.head_forward(out)
        return out[0], {'prelogits': out[1]}

@register_model('linear_seq')
class LinearTaggerModel(TaggerModel):

    @classmethod
    def build_model_decoder(cls, args, dictionary, output_dictionary, embed_tokens):

        kwargs = StructuredLogits._read_and_load_in_args(args)

        decoder = LinearDecoder(
            args,
            output_dictionary,
            embed_tokens,
            **kwargs
        )

        return decoder

    @staticmethod
    def add_args(parser):
        TaggerModel.add_args(parser)
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
            help='dropout probability')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
            help='num decoder layers')
        parser.add_argument('--decoder-norm', action='store_true',
                            help='Use BatchNorm')
        parser.add_argument('--decoder-activation', default='relu', choices=['relu', 'swish', 'tanh', 'penalized_tanh', 'mish'])
        parser.add_argument('--decoder-last-activation', action='store_true',
                            help='Use activation after last hidden linear layer before the output matrix.')
        parser.add_argument('--decoder-output-use-bias', action='store_true',
            help='Use bias in output layer.')
        parser.add_argument('--decoder-output-dim', type=int, default=512)

class LinearDecoderEnsemble(FairseqTaggerDecoder):

    def __init__(self, dictionary, *args, **kwargs):
        super(FairseqTaggerDecoder, self).__init__(dictionary)
        self.decoders = torch.nn.ModuleList(*args, **kwargs)

    def forward(self, tokens, precomputed_embedded):    
        out = 0
        for dec in self.decoders:
            #out *= F.softmax(dec(tokens, precomputed_embedded)[0], -1) # we have negatives in the logits
            out = out + F.softmax(dec(tokens, precomputed_embedded)[0], -1) # we have negatives in the logits
        return out, {}

class LinearTaggerEnsembleModel(LinearTaggerModel):

    @staticmethod
    def make_args_iterator(paths, map_location=None):
        for path in paths:
            yield torch.load(path, map_location=map_location)

    @classmethod
    def build_model(cls, args_it, task):
        """Build a new model instance."""
        args_it = iter(args_it)
        args = next(args_it)
        args_state = args['args']
        embedder = cls.build_model_input(args_state, task.dictionary)
        assert isinstance(embedder, BaseContextualEmbedder)
        args_it = itertools.chain([args], args_it)

        decoders = []
        for args in args_it:
            dec = cls.build_model_decoder(
                args['args'],
                task.dictionary,
                task.output_dictionary,
                FakeInput(embedder.embedding_dim)
            )
            state = collections.OrderedDict()
            for name, param in args['model'].items():
                field, field_of_field_r = name.split('.', 1)
                assert field == 'decoder'
                state[field_of_field_r] = param
            dec.load_state_dict(state)
            decoders.append(dec)

        decoders = LinearDecoderEnsemble(task.dictionary, decoders)
        inst = cls(
            embedder,
            decoders,
            task.kind,
            use_all_hidden=args_state.context_embeddings_use_all_hidden
        )
        return inst


@register_model_architecture('linear_seq', 'linear_seq')
def linear_seq(args):

    args.decoder_norm = getattr(args, 'decoder_norm', False)
    args.decoder_activation = getattr(args, "decoder_activation", 'relu')
    args.decoder_last_relu = getattr(args, "decoder_last_activation", False)
    args.decoder_output_use_bias = getattr(args, "decoder_output_use_bias", False)

    args.context_embeddings_use_all_hidden = getattr(args, "context_embeddings_use_all_hidden", False)
    args.context_embeddings_normalize_embeddings = getattr(args, "context_embeddings_normalize_embeddings", True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.criterion = getattr(args, 'criterion', "weighted_cross_entropy")






