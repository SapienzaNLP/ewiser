from collections import defaultdict
from pathlib import Path
import sys

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from tqdm import tqdm
import numpy as np


from ewiser.fairseq_ext.data.utils import patched_lemma_from_key

from reduce_dims import read_embeddings, write_embeddings, norm

def get_sense(x, dct):
    try:
        o = dct[x]
    except KeyError as e:
        x = x.replace('%3', '%5')
        o = dct.get(x)
        if not o:
            raise e
    return o

def read_mappings(path, use_first=False):
    mappings = defaultdict(set)
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            to, *fr = line.split()
            for x in fr:
                mappings[x.lower()].add(to)
    mappings2 = {}
    for k, v in mappings.items():
        v = list(v)
        i = 0
        while True:
            try:
                sense = patched_lemma_from_key(v[i], wn)
            except WordNetError:
                i += 1
                continue
            break
        senses = {sns.key(): i for i, sns in enumerate(sense.synset().lemmas())}
        senses = sorted(v, key=lambda x: get_sense(x, senses))
        senses = senses[0:1] if use_first else senses
        mappings2[k] = senses
    return mappings2

def calc_centroids(words, matrix, mappings):

    rev_words = {w: i for i, w in enumerate(words)}

    new_words = list(mappings.keys())
    new_matrix = np.zeros((len(new_words), matrix.shape[1]))

    for i_new, w in enumerate(new_words):
        vecs = []
        for orig in mappings[w]:
            try:
                i_orig = rev_words[orig]
            except KeyError:
                continue
            vecs.append(matrix[i_orig])
        centroid = (sum(vecs) / len(vecs)) if vecs else 0
        new_matrix[i_new, :] = centroid

    return new_words, new_matrix

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_embeddings', type=str)
    parser.add_argument('output_embeddings', type=str)
    parser.add_argument('mappings', type=str)
    parser.add_argument('-f', '--use_first', action='store_true')

    args = parser.parse_args()

    words, matrix = read_embeddings(args.input_embeddings)
    mappings = read_mappings(args.mappings, use_first=args.use_first)

    new_words, new_matrix = calc_centroids(words, matrix, mappings)
    matrix = norm(matrix)

    write_embeddings(args.output_embeddings, new_words, new_matrix)

