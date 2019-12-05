#!/usr/bin/env python
import os
import numpy as np
import pickle as pkl
from data import read_signatures
from utils import enumerate_paths
from utils import split_by
from evaluate import evaluate


def cosine_similarity(a, b):
    # Compute the cosine similarity between all vectors in a and b [NxC]
    _a = a / np.sqrt(np.sum(np.square(a), axis=1, keepdims=True))
    _b = b / np.sqrt(np.sum(np.square(b), axis=1, keepdims=True))
    return _a.dot(_b.T)


def train_test_split(person_ids, video_ids, train_to_test_ratio=0.5):
    # Splits the videos of each person to train/test according to the train_to_test_ratio

    # Find borders where person id changes
    sections = np.where(np.diff(person_ids, 1))[0] + 1
    # videos split by person id
    person_videos = np.split(video_ids, sections)
    # Indices split by person id
    frame_indices = np.split(np.arange(len(person_ids)), sections)

    # Split videos train and test according to the train_to_test_ratio
    train_indices = []
    test_indices = []
    for pid, cids, fidx in zip(person_ids, person_videos, frame_indices):
        unique_cids = np.arange(cids[0], cids[-1] + 1)
        is_train = np.random.rand(len(unique_cids)) < train_to_test_ratio
        if not np.any(is_train):
            is_train[0] = True
        is_train_dict = dict(zip(unique_cids, is_train))
        is_train = np.array([is_train_dict[cid] for cid in cids])
        train_indices.append(fidx[is_train])
        test_indices.append(fidx[~ is_train])
    train_indices = np.hstack(train_indices)
    test_indices = np.hstack(test_indices)
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    return train_indices, test_indices


def mean_signatures(signatures, indices):
    # Compute the mean signaures for each set of indices
    mean_signatures = np.vstack([np.mean(signatures[idx], axis=0)
                                 for idx in indices])
    return mean_signatures


def main(sigs_path, train_to_test_ratio=0.5):
    # Read the imagenet signatures from file
    paths, signatures = read_signatures(sigs_path)
    # Enumerate the frame paths based on person and video
    person_ids, video_ids = enumerate_paths(paths)
    # For each person, split his set of videos to train and test
    train_indices, test_indices = train_test_split(person_ids, video_ids,
                                                   train_to_test_ratio)

    # Solution

    # Find the mean signature for each person based on the training set
    train_sigs = split_by(signatures[train_indices], person_ids[train_indices])
    train_sigs = np.vstack([np.mean(ts, axis=0) for ts in train_sigs])

    # Find the mean signature for each test - video and assign its ground-truth person id
    test_sigs = split_by(signatures[test_indices], video_ids[test_indices])
    test_sigs = np.vstack([np.mean(ts, axis=0) for ts in test_sigs])
    # Ground truth labels
    test_labels = np.array([pids[0] for pids in
                            split_by(person_ids[test_indices], video_ids[test_indices])])

    # Predict classes using cosine similarity
    similarity_matrix = cosine_similarity(test_sigs, train_sigs)

    # Crate a submission - a sorted list of predictions, best match on the left.
    ranking = similarity_matrix.argsort(axis=1)
    submission = [line.tolist() for line in ranking[:, :-6:-1]]

    # Compute and display top 1 / 5 accuracies
    evaluate(submission, test_labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Naive solution')
    parser.add_argument(
        '--sigs_path',  help='path for signatures pkl', default='data/signatures.pkl')
    parser.add_argument(
        '--train_to_test_ratio',  help='train to test ratio', type=float, default=0.5)
    args = parser.parse_args()

    main(**vars(args))
