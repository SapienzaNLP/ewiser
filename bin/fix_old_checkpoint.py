import torch
import sys

from ewiser import _import_this

def delparameter(args, pars):
    for p in pars:
        if p in args.__dict__:
            del args.__dict__[p]


def subparameter(args, froms, tos):
    setp = []
    for f, t in zip(froms, tos):
        if f in args.__dict__:
            setattr(args, t, args.__dict__[f])
            setp.append(f)
    delparameter(args, setp)

def submodelparameter(parameters, froms, tos):
    for f, t in zip(froms, tos):
        if f in parameters:
            tmp = parameters[f]
            del parameters[f]
            parameters[t] = tmp

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Utility script to convert legacy checkpoints to the format required by the current version of the library.')
    parser.add_argument('old', help='Path to old checkpoint.')
    parser.add_argument('new', help='Path to save fixed checkpoint in.')
    args = parser.parse_args()

    dels = [
        "decoder_structured_logits_non_zero_classes",
        "decoder_syntagmatic_window",
        "decoder_syntagmatic_arcs",
        "decoder_syntagmatic_arcs_trainable",
        "decoder_structured_logits_parametrize_self_loops",
        "context_embeddings_xlm_checkpoint",
        "context_embeddings_xlm_bpe_codes",
        "context_embeddings_xlm_bpe_vocab",
        "context_embeddings_xlnet_model",
        "context_embeddings_gpt2_model gpt2",
        "context_embeddings_elmo_options",
        "context_embeddings_elmo_weights",
        "context_embeddings_elmot_path",
        "context_embeddings_qbert_checkpoint",
        "context_embeddings_flair_forward",
        "context_embeddings_flair_backward",
        "context_embeddings_flair_embeddings",
    ]

    froms, tos = zip(
        ('decoder_paradigmatic_arcs', 'decoder_structured_logits_edgelists'),
        ('decoder_paradigmatic_arcs_trainable', 'decoder_structured_logits_trainable'),
    )

    fromsp, tosp = zip(
        ('decoder.structured_logits.paradigmatic_adjacency_pars.0', 'decoder.structured_logits.adjacency_pars.0'),
        ('decoder.structured_logits.paradigmatic_adjacency_pars.1', 'decoder.structured_logits.adjacency_pars.1'),
        ('decoder.structured_logits.paradigmatic_adjacency_pars.2', 'decoder.structured_logits.adjacency_pars.2'),
    )

    data = torch.load(args.old, map_location='cpu')
    delparameter(data['args'], dels)
    subparameter(data['args'], froms, tos)

    submodelparameter(data['model'], fromsp, tosp)

    for k, v in data['args'].__dict__.items():
        print(k, v)

    torch.save(data, sys.argv[2])