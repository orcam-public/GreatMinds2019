from __future__ import print_function

import torch
import numpy as np
from data import Images
from data import read_signatures
from utils import enumerate_paths
from utils import split_by
from torch.utils.tensorboard import SummaryWriter


def main(sigs_path, images_path, samples_per_person=16):
    # Read the imagenet signatures from file
    paths, signatures = read_signatures(sigs_path)
    # Enumerate the frame paths based on person and video
    person_ids, video_ids = enumerate_paths(paths)
    # Sample "samples_per_person" images from each person
    sampled_indices = [pid for pp in split_by(range(len(paths)), person_ids)
                       for pid in sorted(np.random.choice(pp, samples_per_person).tolist())]
    sampled_paths = [paths[idx] for idx in sampled_indices]
    sampled_labels = np.mgrid[:len(sampled_indices),
                              :samples_per_person][0].ravel()
    # Get images of sampled data points
    with Images(images_path) as images:
        sampled_images = [images[path] for path in sampled_paths]
    sampled_images = np.concatenate([sampled_images]).transpose([0, 3, 1, 2])
    # Get normalized signatures of sampled data points
    sampled_sigs = signatures[sampled_indices]
    sampled_sigs /= np.sqrt(np.sum(np.square(sampled_sigs),
                                   axis=1, keepdims=True))
    # Write data to tensorboard projector
    writer = SummaryWriter()
    meta_data = [sp.split('/')[0] for sp in sampled_paths]
    label_img = torch.from_numpy(sampled_images).float() / 255
    writer.add_embedding(torch.from_numpy(sampled_sigs),
                         metadata=meta_data,
                         label_img=label_img)
    print('Visualization ready')
    print('run: \t tensorboard --logdir=runs')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data Visualization')
    parser.add_argument(
        '--sigs_path',  help='path for signatures pkl', default='signatures.pkl')
    parser.add_argument(
        '--images_path',  help='path for images tar', default='images.tar')
    parser.add_argument(
        '--samples_per_person',  help='samples per person to display', type=int, default=16)
    args = parser.parse_args()

    main(**vars(args))
