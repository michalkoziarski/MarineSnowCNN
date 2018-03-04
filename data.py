import zipfile
import logging
import imageio
import numpy as np

from pathlib import Path
from urllib.request import urlretrieve


DATA_PATH = Path(__file__).parent / 'data'
DATA_URL = 'https://www.dropbox.com/s/nlowvsi0nnd4xrh/AGH_MSD_v2.zip?dl=0'
ARCHIVE_PATH = DATA_PATH / 'AGH_MSD_v2.zip'


def _truncate_frames(frames):
    truncated_frames = np.empty(frames.shape + (1,), dtype=frames.dtype)

    for i in range(len(frames)):
        frames[i, frames[i] < 0.5] = 0.0
        frames[i, frames[i] >= 0.5] = 1.0

    return truncated_frames


def _load_frames(partition_name, temporal_width):
    partition_path = DATA_PATH / partition_name
    original_path = partition_path / 'Original'
    ground_truth_path = partition_path / 'GroundTruth'

    if not partition_path.exists():
        raise ValueError('Unrecognized partition "%s".' % partition_name)

    ground_truth_frame_ids = [int(path.stem.replace('frame', '')) for path in ground_truth_path.glob('*.jpg')]
    ground_truth_frame_ids.sort()

    left_extension = list(range(ground_truth_frame_ids[0] - temporal_width // 2, ground_truth_frame_ids[0]))
    right_extension = list(range(ground_truth_frame_ids[-1] + 1, ground_truth_frame_ids[-1] + 1 + temporal_width // 2))
    original_frame_ids = left_extension + ground_truth_frame_ids + right_extension

    original_frames = np.array([imageio.imread(str(original_path / ('frame%d.jpg' % i))) / 255
                                for i in original_frame_ids])
    ground_truth_frames = np.array([imageio.imread(str(ground_truth_path / ('frame%d.jpg' % i))) / 255
                                    for i in ground_truth_frame_ids])

    ground_truth_frames = _truncate_frames(ground_truth_frames)

    return original_frames, ground_truth_frames


def _download_if_necessary():
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(exist_ok=True)

    if not ARCHIVE_PATH.exists():
        logging.info('Downloading data archive...')

        urlretrieve(DATA_URL, ARCHIVE_PATH)

        with zipfile.ZipFile(str(ARCHIVE_PATH)) as f:
            logging.info('Extracting data archive...')

            f.extractall(DATA_PATH)


class PatchDataset:
    def __init__(self, partitions, batch_size=64, temporal_patch_size=3, spatial_patch_size=40, spatial_stride=20):
        self.partitions = partitions
        self.batch_size = batch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.spatial_stride = spatial_stride
        self.length = 0
        self.current_batch_index = 0
        self.inputs = None
        self.outputs = None

        _download_if_necessary()

        logging.info('Calculating necessary memory...')

        for partition_name in partitions:
            original_frames, ground_truth_frames = _load_frames(partition_name, temporal_patch_size)

            logging.info('Found %d original and %d ground truth frames for partition "%s".' % (len(original_frames),
                                                                                               len(ground_truth_frames),
                                                                                               partition_name))

            shape = ground_truth_frames.shape

            for t in range(shape[0]):
                for x in range(0, shape[1] - spatial_stride, spatial_patch_size - spatial_stride):
                    for y in range(0, shape[2] - spatial_stride, spatial_patch_size - spatial_stride):
                        self.length += 4

        logging.info('Allocating memory...')

        self.inputs = np.empty([self.length, temporal_patch_size, spatial_patch_size, spatial_patch_size, 3],
                               dtype=np.float32)
        self.outputs = np.empty([self.length, spatial_patch_size, spatial_patch_size, 1],
                                dtype=np.float32)
        self.indices = list(range(self.length))

        logging.info('Loading patches...')

        current_patch_index = 0

        for partition_name in partitions:
            original_frames, ground_truth_frames = _load_frames(partition_name, temporal_patch_size)

            shape = ground_truth_frames.shape

            for t in range(shape[0]):
                for x in range(0, shape[1] - spatial_stride, spatial_patch_size - spatial_stride):
                    for y in range(0, shape[2] - spatial_stride, spatial_patch_size - spatial_stride):
                        original_patch = original_frames[t:(t + temporal_patch_size),
                                                         x:(x + spatial_patch_size),
                                                         y:(y + spatial_patch_size)]

                        ground_truth_patch = ground_truth_frames[t,
                                                                 x:(x + spatial_patch_size),
                                                                 y:(y + spatial_patch_size)]

                        for n_rotations in range(2):
                            for flip_xy in [False, True]:
                                augmented_original_patch = original_patch.copy()
                                augmented_ground_truth_patch = ground_truth_patch.copy()

                                augmented_original_patch = np.rot90(augmented_original_patch,
                                                                    k=n_rotations, axes=(1, 2))
                                augmented_ground_truth_patch = np.rot90(augmented_ground_truth_patch,
                                                                        k=n_rotations, axes=(0, 1))

                                if flip_xy:
                                    augmented_original_patch = augmented_original_patch[:, ::-1]
                                    augmented_ground_truth_patch = augmented_ground_truth_patch[::-1]

                                self.inputs[current_patch_index] = augmented_original_patch
                                self.outputs[current_patch_index] = augmented_ground_truth_patch

                                current_patch_index += 1

    def batch(self):
        batch_indices = self.indices[self.current_batch_index:(self.current_batch_index + self.batch_size)]

        inputs = self.inputs[batch_indices]
        outputs = self.outputs[batch_indices]

        self.current_batch_index += self.batch_size

        if self.current_batch_index >= self.length:
            self.current_batch_index = 0

        return inputs, outputs

    def shuffle(self):
        np.random.shuffle(self.indices)


class ImageDataset:
    def __init__(self, partitions, temporal_width):
        self.partitions = partitions
        self.temporal_width = temporal_width
        self.inputs = []
        self.outputs = []
        self.length = 0
        self.current_fetch_index = 0

        _download_if_necessary()

        logging.info('Loading frames...')

        for partition_name in partitions:
            original_frames, ground_truth_frames = _load_frames(partition_name, temporal_width)

            for i in range(len(ground_truth_frames)):
                self.inputs.append(original_frames[i:(i + temporal_width)])
                self.outputs.append(ground_truth_frames[i])

                self.length += 1

    def fetch(self):
        fetch_input = self.inputs[self.current_fetch_index]
        fetch_output = self.outputs[self.current_fetch_index]

        self.current_fetch_index += 1

        if self.current_fetch_index >= self.length:
            self.current_fetch_index = 0

        return fetch_input, fetch_output
