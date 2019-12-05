from __future__ import print_function

import os
import cv2
import tarfile
import numpy as np
import pickle as pkl
from utils import parse_tarinfo


class Images(object):
    # A class for easy and fast reading of images packed in a tar file
    def __init__(self, path, index_path=None):
        self.path = path
        if index_path is None:
            # index file is the same as tar path but  .pkl
            index_path = path[:-3] + 'pkl'
        if not os.path.exists(index_path):
            print('Indexing tar file, this could take a few minutes...')
            self._tar_index = self._index_tar(path)
            print('done')
            # Save index file
            with open(index_path, 'wb') as fid:
                pkl.dump(self._tar_index, fid)
        else:
            with open(index_path, 'rb') as fid:
                self._tar_index = pkl.load(fid)
        self.index_path = index_path
        # Open the tar file
        self.fid = open(path, 'rb')
        # Get its size for later checking the indexing validity
        self.fid.seek(0, 2)
        self.tar_size = self.fid.tell()
        # save a sorted list of the tar file paths (keys)
        self.keys = sorted(self._tar_index.keys())

    @staticmethod
    def _index_tar(path):
        # Build a dictionary with the locations of all data points
        tar_index = {}
        with tarfile.TarFile(path, "r") as tar:
            for tarinfo in tar:
                if tarinfo.isfile():
                    offsets_and_size = (
                        tarinfo.offset, tarinfo.offset_data, tarinfo.size)
                    tar_index[tarinfo.name] = offsets_and_size
        return tar_index

    @staticmethod
    def _decode_image(buff):
        # Decode an image buffer from memory
        buff_array = np.asarray(bytearray(buff), dtype='uint8')
        image = cv2.imdecode(buff_array, cv2.IMREAD_UNCHANGED)
        return image

    def __len__(self):
        return len(self._tar_index)

    @property
    def paths(self):
        return self.keys

    def _getitem(self, item):
        # A private _getitem for better readability
        # If item is an index, replace with the path at that index
        if isinstance(item, int):
            item = self.keys[item]
        # Grab an image buffer based on its path and decode it
        offset, data_offset, size = self._tar_index[item]
        # Go to start of record
        self.fid.seek(offset)
        # Check indexing validty
        header_size = data_offset - offset  # should always be 512
        tarinfo = parse_tarinfo(self.fid.read(header_size))
        if tarinfo.path != item:
            raise tarfile.InvalidHeaderError
        buff = self.fid.read(size)
        image = self._decode_image(buff)[:, :, ::-1]
        return image

    def __getitem__(self, item):
        try:
            image = self._getitem(item)
        except (tarfile.InvalidHeaderError, tarfile.TruncatedHeaderError, tarfile.EmptyHeaderError):
            error_str = 'Index file "{}" does not match tarfile "{}". Remove the index file and try again.'
            raise IOError(error_str.format(self.index_path, self.path))

        return image

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.fid.close()


def compatible_load(path):
    # pickle loading compatible for pyton 2/3
    data = None
    with open(path, 'rb') as fid:
        try:
            data = pkl.load(fid)
        except UnicodeDecodeError:
            # Python 3 compatability
            fid.seek(0)
            data = pkl.load(fid, encoding='latin1')
    return data


def read_pose(pose_path):
    # Read the pose points from file
    data = compatible_load(pose_path)
    keypoints = data['keypoints']
    scores = data['scores']
    paths = data['paths']
    return paths, keypoints, scores


def read_signatures(sigs_path):
    # Read the imagenet signatures from file
    data = compatible_load(sigs_path)
    signatures = data['signatures']
    paths = data['paths']
    return paths, signatures
