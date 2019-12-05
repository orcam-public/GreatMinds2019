import os
import sys
import tarfile
import numpy as np

FSENCODING = sys.getfilesystemencoding()


def enumerate_paths(paths):
    # Extract sequences/videos/people from the frame-paths
    sequences = [os.path.dirname(p) for p in paths]
    videos = [os.path.dirname(s) for s in sequences]
    people = [os.path.dirname(c) for c in videos]

    # Enumerate the frames based on videos and people
    unique_videos, video_ids = np.unique(videos, return_inverse=True)
    unique_people, person_ids = np.unique(people, return_inverse=True)
    return person_ids, video_ids


def split_by(data, indices):
    # Split data based on a numpy array of sorted indices
    sections = np.where(np.diff(indices))[0] + 1
    split_data = np.split(data, sections)
    return split_data


def parse_tarinfo(buff):
    # Get a version-compatible tarinfo parser
    if not hasattr(parse_tarinfo, 'defaultargs'):
        # Determine version once on first call
        dummy_header = tarfile.TarInfo().tobuf()
        try:
            _ = tarfile.TarInfo.frombuf(dummy_header)
            parse_tarinfo.defaultargs = False
        except TypeError:
            parse_tarinfo.defaultargs = True
    if parse_tarinfo.defaultargs:
        # Python 3
        return tarfile.TarInfo.frombuf(buff, FSENCODING, 'surrogateescape')
    else:
        # Python 2
        return tarfile.TarInfo.frombuf(buff)
