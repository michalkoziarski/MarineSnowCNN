import zipfile
import logging
import imageio
import numpy as np

from pathlib import Path
from urllib.request import urlretrieve


DATA_PATH = Path(__file__).parent / 'data'
DATA_URL = 'http://home.agh.edu.pl/~cyganek/AGH_MSD.zip'
ARCHIVE_PATH = DATA_PATH / 'AGH_MSD.zip'


class AnnotatedDataset:
    def __init__(self, partitions, batch_size=64, patch_size=(5, 40, 40, 3), stride=20):
        self.partitions = partitions
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
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

            original_path = partition_path / 'Oryginal'

            frame_ids = [int(path.stem.replace('frame', '')) for path in original_path.glob('*.jpg')]
            frame_ids.sort()

            original_frames = np.array([imageio.imread(str(original_path / ('frame%d.jpg' % i))) / 255
                                        for i in frame_ids])

            shape = original_frames.shape

            for t in range(shape[0] - (patch_size[0] - 1)):
                for x in range(0, shape[1] - stride, patch_size[1] - stride):
                    for y in range(0, shape[2] - stride, patch_size[2] - stride):
                        self.length += 8

        logging.info('Allocating memory...')

        self.inputs = np.empty([self.length] + list(self.patch_size), dtype=np.float32)
        self.outputs = np.empty([self.length] + list(self.patch_size[1:]), dtype=np.float32)

        current_patch_index = 0

        for partition_name in partitions:
            logging.info('Loading "%s" partition...' % partition_name)

            partition_path = DATA_PATH / partition_name

            if not partition_path.exists():
                raise ValueError('Unrecognized partition "%s".' % partition_name)

            original_path = partition_path / 'Oryginal'
            ground_truth_path = partition_path / 'RealGroundTruth'

            frame_ids = [int(path.stem.replace('frame', '')) for path in original_path.glob('*.jpg')]
            frame_ids.sort()

            original_frames = np.array([imageio.imread(str(original_path / ('frame%d.jpg' % i))) / 255
                                        for i in frame_ids])
            ground_truth_frames = np.array([imageio.imread(str(ground_truth_path / ('frame%d.jpg' % i))) / 255
                                            for i in frame_ids])

            shape = original_frames.shape

            for t in range(shape[0] - (patch_size[0] - 1)):
                for x in range(0, shape[1] - stride, patch_size[1] - stride):
                    for y in range(0, shape[2] - stride, patch_size[2] - stride):
                        original_patch = original_frames[t:(t + patch_size[0]),
                                                         x:(x + patch_size[1]),
                                                         y:(y + patch_size[2])]

                        ground_truth_patch = ground_truth_frames[t + patch_size[0] // 2,
                                                                 x:(x + patch_size[1]),
                                                                 y:(y + patch_size[2])]

                        for n_rotations in range(2):
                            for flip_xy in [False, True]:
                                for flip_t in [False, True]:
                                    augmented_original_patch = original_patch.copy()
                                    augmented_ground_truth_patch = ground_truth_patch.copy()

                                    augmented_original_patch = np.rot90(augmented_original_patch,
                                                                        k=n_rotations, axes=(1, 2))
                                    augmented_ground_truth_patch = np.rot90(augmented_ground_truth_patch,
                                                                            k=n_rotations, axes=(0, 1))

                                    if flip_xy:
                                        augmented_original_patch = augmented_original_patch[:, ::-1]
                                        augmented_ground_truth_patch = augmented_ground_truth_patch[::-1]

                                    if flip_t:
                                        augmented_original_patch = augmented_original_patch[::-1]

                                    self.inputs[current_patch_index] = augmented_original_patch
                                    self.outputs[current_patch_index] = augmented_ground_truth_patch

                                    current_patch_index += 1

    def batch(self):
        inputs = self.inputs[self.current_batch_index:(self.current_batch_index + self.batch_size)]
        outputs = self.outputs[self.current_batch_index:(self.current_batch_index + self.batch_size)]

        self.current_batch_index += self.batch_size

        if self.current_batch_index >= self.length:
            self.current_batch_index = 0

        return inputs, outputs

    def shuffle(self):
        indices = list(range(self.length))
        np.random.shuffle(indices)

        self.inputs = self.inputs[indices]
        self.outputs = self.outputs[indices]
