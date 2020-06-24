from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

def read_embeddings(path):
    data = []
    words = []

    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            pieces = line.split()
            if len(pieces) <= 3:
                continue
            else:
                token_type, *floats = pieces
                floats = [float(x) for x in floats]
                data.append(floats)
                words.append(token_type)

    return words, np.array(data)

def norm(matrix):
    norm = np.sum(matrix ** 2, axis=1, keepdims=True) ** 0.5
    matrix /= norm
    return matrix

def reduce_dims(target_dims, matrix):
    matrix = norm(matrix)
    svd = TruncatedSVD(n_components=target_dims, random_state=42)
    matrix = svd.fit_transform(matrix)
    matrix = norm(matrix)
    return matrix, svd

def write_embeddings(path, words, matrix):

    with Path(path).open('w') as f:
        for w, vec in sorted(zip(words, matrix), key=lambda x: x[0]):
            pieces = [w] + [str(x) for x in vec.tolist()]
            f.write(' '.join(pieces) + '\n')

if __name__ == '__main__':

    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-d', '--dims', type=int, default=512)

    args = parser.parse_args()

    words, matrix = read_embeddings(args.input)
    matrix, svd = reduce_dims(args.dims, matrix)

    #with open(args.input + '-reducer.pkl', 'wb') as b:
    #    pickle.dump(svd, b)

    write_embeddings(args.output, words, matrix)





