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

        if not DATA_PATH.exists():
            DATA_PATH.mkdir()

        if not ARCHIVE_PATH.exists():
            logging.info('Downloading data archive...')

            urlretrieve(DATA_URL, ARCHIVE_PATH)

            with zipfile.ZipFile(ARCHIVE_PATH) as f:
                logging.info('Extracting data archive...')

                f.extractall(DATA_PATH)

        original_patches = []
        ground_truth_patches = []

        for partition_name in partitions:
            logging.info('Loading "%s" partition...' % partition_name)

            partition_path = DATA_PATH / partition_name

            if not partition_path.exists():
                raise ValueError('Unrecognized partition "%s".' % partition_name)

            original_path = partition_path / 'Oryginal'
            ground_truth_path = partition_path / 'RealGroundTruth'

            frame_ids = [int(path.stem.replace('frame', '')) for path in original_path.glob('*.jpg')]
            frame_ids.sort()

            original_frames = np.array([imageio.imread(str(original_path / ('frame%d.jpg' % i)))
                                        for i in frame_ids])
            ground_truth_frames = np.array([imageio.imread(str(ground_truth_path / ('frame%d.jpg' % i)))
                                            for i in frame_ids])

            shape = original_frames.shape

            for t in range(shape[0] - (patch_size[0] - 1)):
                for x in range(0, shape[1], patch_size[1] - stride):
                    for y in range(0, shape[2], patch_size[2] - stride):
                        original_patch = original_frames[t:(t + patch_size[0]),
                                                         x:(x + patch_size[1]),
                                                         y:(y + patch_size[2])].copy()

                        ground_truth_patch = ground_truth_frames[t:(t + patch_size[0]),
                                                                 x:(x + patch_size[1]),
                                                                 y:(y + patch_size[2])].copy()

                        original_patches.append(original_patch)
                        ground_truth_patches.append(ground_truth_patch)

        self.length = len(original_patches)
        self.current_index = 0
        self.inputs = np.array(original_patches)
        self.outputs = np.array(ground_truth_patches)

    def batch(self):
        inputs = self.inputs[self.current_index:(self.current_index + self.batch_size)]
        outputs = self.outputs[self.current_index:(self.current_index + self.batch_size)]

        self.current_index += self.batch_size

        if self.current_index >= self.length:
            self.current_index = 0

        return inputs, outputs

    def shuffle(self):
        indices = list(range(self.length))
        np.random.shuffle(indices)

        self.inputs = self.inputs[indices]
        self.outputs = self.outputs[indices]
