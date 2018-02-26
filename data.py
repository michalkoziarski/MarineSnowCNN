import zipfile
import logging
import imageio
import numpy as np

from pathlib import Path
from urllib.request import urlretrieve


DATA_PATH = Path(__file__).parent / 'data'
DATA_URL = 'https://www.dropbox.com/s/n6cux3vepkg7824/AGH_MSD.zip'
ARCHIVE_PATH = DATA_PATH / 'AGH_MSD.zip'


class AnnotatedDataset:
    def __init__(self, partitions, batch_size=64, temporal_patch_size=3, spatial_patch_size=40, spatial_stride=20,
                 annotation_type='mask'):
        assert annotation_type in ['mask', 'filtered']

        self.partitions = partitions
        self.batch_size = batch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.spatial_stride = spatial_stride
        self.annotation_type = annotation_type
        self.length = 0
        self.current_batch_index = 0
        self.inputs = None
        self.outputs = None

        if not DATA_PATH.exists():
            DATA_PATH.mkdir()

        if not ARCHIVE_PATH.exists():
            logging.info('Downloading data archive...')

            urlretrieve(DATA_URL, ARCHIVE_PATH)

            with zipfile.ZipFile(str(ARCHIVE_PATH)) as f:
                logging.info('Extracting data archive...')

                f.extractall(DATA_PATH)

        logging.info('Calculating necessary memory...')

        for partition_name in partitions:
            partition_path = DATA_PATH / partition_name

            if not partition_path.exists():
                raise ValueError('Unrecognized partition "%s".' % partition_name)

            original_path = partition_path / 'Original'

            frame_ids = [int(path.stem.replace('frame', '')) for path in original_path.glob('*.jpg')]
            frame_ids.sort()

            original_frames = np.array([imageio.imread(str(original_path / ('frame%d.jpg' % i))) / 255
                                        for i in frame_ids])

            shape = original_frames.shape

            for t in range(shape[0] - (temporal_patch_size - 1)):
                for x in range(0, shape[1] - spatial_stride, spatial_patch_size - spatial_stride):
                    for y in range(0, shape[2] - spatial_stride, spatial_patch_size - spatial_stride):
                        self.length += 4

        logging.info('Allocating memory...')

        if annotation_type == 'mask':
            n_output_channels = 1
        elif annotation_type == 'filtered':
            n_output_channels = 3
        else:
            raise NotImplementedError

        self.inputs = np.empty([self.length, temporal_patch_size, spatial_patch_size, spatial_patch_size, 3],
                               dtype=np.float32)
        self.outputs = np.empty([self.length, spatial_patch_size, spatial_patch_size, n_output_channels],
                                dtype=np.float32)
        self.indices = list(range(self.length))

        current_patch_index = 0

        for partition_name in partitions:
            logging.info('Loading "%s" partition...' % partition_name)

            partition_path = DATA_PATH / partition_name

            if not partition_path.exists():
                raise ValueError('Unrecognized partition "%s".' % partition_name)

            original_path = partition_path / 'Original'

            if self.annotation_type == 'mask':
                ground_truth_path = partition_path / 'BlackGroundTruth'
            elif self.annotation_type == 'filtered':
                ground_truth_path = partition_path / 'Filtered'
            else:
                raise NotImplementedError

            frame_ids = [int(path.stem.replace('frame', '')) for path in original_path.glob('*.jpg')]
            frame_ids.sort()

            original_frames = np.array([imageio.imread(str(original_path / ('frame%d.jpg' % i))) / 255
                                        for i in frame_ids])
            ground_truth_frames = np.array([imageio.imread(str(ground_truth_path / ('frame%d.jpg' % i))) / 255
                                            for i in frame_ids])

            if self.annotation_type == 'mask':
                truncated_ground_truth_frames = np.empty(ground_truth_frames.shape[:-1] + (1, ),
                                                         dtype=ground_truth_frames.dtype)

                for i in range(len(ground_truth_frames)):
                    ground_truth_frames[i, ground_truth_frames[i] < 0.5] = 0.0
                    ground_truth_frames[i, ground_truth_frames[i] >= 0.5] = 1.0

                    truncated_ground_truth_frames[i] = np.dstack([np.max(ground_truth_frames[i], axis=2)])

                ground_truth_frames = truncated_ground_truth_frames

            shape = original_frames.shape

            for t in range(shape[0] - (temporal_patch_size - 1)):
                for x in range(0, shape[1] - spatial_stride, spatial_patch_size - spatial_stride):
                    for y in range(0, shape[2] - spatial_stride, spatial_patch_size - spatial_stride):
                        original_patch = original_frames[t:(t + temporal_patch_size),
                                                         x:(x + spatial_patch_size),
                                                         y:(y + spatial_patch_size)]

                        ground_truth_patch = ground_truth_frames[t + temporal_patch_size // 2,
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
