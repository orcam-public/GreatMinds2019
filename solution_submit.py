#!/usr/bin/env python
import os
import numpy as np
import pickle as pkl
from data import read_signatures
from utils import enumerate_paths
from utils import split_by
from evaluate import evaluate
from evaluate import submit


def cosine_similarity(a, b):
    # Compute the cosine similarity between all vectors in a and b [NxC]
    _a = a / np.sqrt(np.sum(np.square(a), axis=1, keepdims=True))
    _b = b / np.sqrt(np.sum(np.square(b), axis=1, keepdims=True))
    return _a.dot(_b.T)


def mean_signatures(signatures, indices):
    # Compute the mean signaures for each set of indices
    mean_signatures = np.vstack([np.mean(signatures[idx], axis=0)
                                 for idx in indices])
    return mean_signatures


def main(sigs_train, sigs_test):
    # Read the imagenet signatures from file
    paths_train, train_sigs = read_signatures(sigs_train)
    paths_test, test_sigs = read_signatures(sigs_test)

    # Solution

    # Find the mean signature for each person based on the training set
    person_ids = np.array([int(p.split('/')[0][7:]) for p in paths_train])
    train_person_sigs = split_by(train_sigs, person_ids)
    train_person_sigs = np.vstack([np.mean(ts, axis=0)
                                   for ts in train_person_sigs])

    # Find the mean signature for each test sequence
    seq_ids = np.array([int(p.split('/')[0][4:]) for p in paths_test])
    test_seq_sigs = split_by(test_sigs, seq_ids)
    test_seq_sigs = np.vstack([np.mean(ts, axis=0) for ts in test_seq_sigs])

    # Predict classes using cosine similarity
    similarity_matrix = cosine_similarity(test_seq_sigs, train_person_sigs)

    # Crate a submission - a sorted list of predictions, best match on the left.
    ranking = similarity_matrix.argsort(axis=1)
    submission = [line.tolist() for line in ranking[:, :-6:-1]]

    # submit to server, print reply (-1 means something is wrong)
    print(submit('naive', submission))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Naive solution')
    parser.add_argument(
        '--sigs_train',  help='path for train signatures pkl', default='data/signatures.pkl')
    parser.add_argument(
        '--sigs_test',  help='path for test signatures pkl', default='data/signatures_test.pkl')
    args = parser.parse_args()

    main(**vars(args))
