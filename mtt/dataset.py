import io
import os
import pathlib
import pickle
import random

import av
import librosa
import torchaudio
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DistributedSampler, WeightedRandomSampler

import torch
from ba3l.ingredients.datasets import Dataset
from sacred.config import DynamicIngredient, CMD
from scipy.signal import convolve
import numpy as np
from helpers.audiodatasets import PreprocessDataset
import h5py


#$TMPDIR
dataset = Dataset('mtt')


@dataset.config
def default_config():
    name = 'mtt'  # dataset name
    normalize = False  # normalize dataset
    subsample = False  # subsample squares from the dataset
    roll = False  # apply roll augmentation
    fold = 1
    base_dir = "/home/palonso/data/magnatagatune-melspectrograms/"  # base directory of the dataset, change it or make a link

    val_groundtruth = "mtt/groundtruth-validation.pk"
    train_groundtruth = "mtt/groundtruth-train.pk"
    test_groundtruth = "mtt/groundtruth-test.pk"
    num_of_classes = 50


class MTTDataset(TorchDataset):
    def __init__(
        self,
        groundtruth_file, base_dir="",
        sample_rate=16000,
        classes_num=50,
        clip_length=10,
        augment=False,
        hop_size=256,
        n_bands=96
    ):
        """
        Reads the mel spectrogram chunks with numpy and returns a fixed length mel-spectrogram patch
        """

        self.base_dir = base_dir
        with open(groundtruth_file, "rb") as gf:
            self.groundtruth = pickle.load(gf)
        self.filenames = {i: filename for i, filename in enumerate(list(self.groundtruth.keys()))}
        self.length = len(self.groundtruth)
        self.sample_rate = sample_rate
        self.dataset_file = None  # lazy init
        self.clip_length = clip_length
        self.classes_num = classes_num
        self.augment = augment
        if augment:
            print(f"Will agument data from {groundtruth_file}")

        self.melspectrogram_size = clip_length * sample_rate // hop_size
        self.n_bands = n_bands

    def __len__(self):
        return self.length

    def __del__(self):
        if self.dataset_file is not None:
            self.dataset_file.close()
            self.dataset_file = None

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.

        Args:
          index: int
        Returns:
          data_dict: {
            'melspectrogram': (bands, timestamps,),
            'filename_name': str,
            'target': (classes_num,)}
        """

        filename = self.filenames[index]
        target = self.groundtruth[filename].astype("float16")

        melspectrogram_file = pathlib.Path(self.base_dir, filename)
        melspectrogram = self.load_melspectrogram(melspectrogram_file)


        return melspectrogram, filename, target

    def load_melspectrogram(self, melspectrogram_file: pathlib.Path, offset: int= None):
        frames_num = melspectrogram_file.stat().st_size // (2 * self.n_bands)  # each float16 has 2 bytes

        if not offset:
            max_frame = frames_num - self.melspectrogram_size
            offset = random.randint(0, max_frame)

        # offset: idx * bands * bytes per float
        offset_bytes = offset * self.n_bands * 2
    
        skip_frames = max(offset + self.melspectrogram_size - frames_num, 0)
        frames_to_read = self.melspectrogram_size - skip_frames

        fp = np.memmap(melspectrogram_file, dtype='float16', mode='r',
                       shape=(frames_to_read, self.n_bands), offset=offset_bytes)

        # put the data in a numpy ndarray
        melspectrogram = np.array(fp, dtype='float16')

        if frames_to_read < self.melspectrogram_size:
            padding_size = self.melspectrogram_size - frames_to_read
            melspectrogram = np.vstack([melspectrogram, np.zeros([padding_size, self.n_bands], dtype="float16")])
            melspectrogram = np.roll(melspectrogram, padding_size // 2, axis=0)  # center the padding

        del fp

        # transpose, PaSST expects dims as [b,e,f,t]
        melspectrogram = melspectrogram.T
        melspectrogram = np.expand_dims(melspectrogram, 0)

        return melspectrogram


class MTTDatasetExhaustive(MTTDataset):
    def __init__(
        self,
        groundtruth_file,
        base_dir="",
        sample_rate=16000,
        classes_num=50,
        clip_length=10,
        augment=False,
        hop_size=256,
        n_bands=96,
    ):
        """
        Reads the mel spectrogram chunks with numpy and returns a fixed length mel-spectrogram patch
        """
        super().__init__(
            groundtruth_file,
            base_dir=base_dir,
            sample_rate=sample_rate,
            classes_num=classes_num,
            clip_length=clip_length,
            augment=augment,
            hop_size=hop_size,
            n_bands=n_bands,
        )

        filenames = []
        for filename in self.filenames.values():
            melspectrogram_file = pathlib.Path(self.base_dir, filename)
            frames_num = melspectrogram_file.stat().st_size // (2 * self.n_bands)  # each float16 has 2 bytes
            # do a last patch up to 20% zero-pad
            n_patches = int((frames_num * 1.2) // self.melspectrogram_size)
            # filenames is a tuple (filename, offset)
            filenames.extend([(filename, i * self.melspectrogram_size) for i in range(n_patches)])

        self.filenames_with_patch = dict(zip(range(len(filenames)), filenames))
        self.length = len(self.filenames_with_patch)


    def __getitem__(self, index):
        """Load waveform and target of an audio clip.

        Args:
          index: int
        Returns:
          data_dict: {
            'melspectrogram': (bands, timestamps,),
            'filename_name': (str, offset),
            'target': (classes_num,)}
        """

        filename, offset = self.filenames_with_patch[index]
        target = self.groundtruth[filename].astype("float16")

        melspectrogram_file = pathlib.Path(self.base_dir, filename)
        melspectrogram = self.load_melspectrogram(melspectrogram_file, offset)

        return melspectrogram, filename, target

@dataset.command
def get_train_set(train_groundtruth):
    ds = MTTDataset(train_groundtruth)
    return ds


@dataset.command
def get_ft_cls_balanced_sample_weights(train_groundtruth, num_of_classes,
                                       sample_weight_offset=100, sample_weight_sum=True):
    """
    :return: float tenosr of shape len(full_training_set) representing the weights of each sample.
    """
    # the order of balanced_train_hdf5,unbalanced_train_hdf5 is important.
    # should match get_full_training_set
    all_y = []
    for mspec_file in [train_groundtruth]:
        with open(mspec_file, 'rb') as dataset_file:
            dataset_data = pickle.load(dataset_file)
            all_y.extend(list(dataset_data.values()))
    all_y = np.array(all_y)
    all_y = torch.as_tensor(all_y)
    per_class = all_y.long().sum(0).float().reshape(1, -1)  # frequencies per class

    per_class = sample_weight_offset + per_class  # offset low freq classes
    if sample_weight_offset > 0:
        print(f"Warning: sample_weight_offset={sample_weight_offset} minnow={per_class.min()}")
    per_class_weights = 1000. / per_class
    all_weight = all_y * per_class_weights
    # print(all_weight.shape)
    # print(all_weight[1510])
    if sample_weight_sum:
        print("\nsample_weight_sum\n")
        all_weight = all_weight.sum(dim=1)
    else:
        all_weight, _ = all_weight.max(dim=1)
    # print(all_weight.shape)
    # print(all_weight[1510])
    return all_weight


@dataset.command
def get_ft_weighted_sampler(samples_weights=CMD(".get_ft_cls_balanced_sample_weights"),
                            epoch_len=600000, sampler_replace=False):
    num_nodes = int(os.environ.get('num_nodes', 1))
    ddp = int(os.environ.get('DDP', 1))
    num_nodes = max(ddp, num_nodes)
    print("num_nodes= ", num_nodes)
    rank = int(os.environ.get('NODE_RANK', 0))
    return DistributedSamplerWrapper(sampler=WeightedRandomSampler(samples_weights,
                                                                   num_samples=epoch_len, replacement=sampler_replace),
                                     dataset=range(epoch_len),
                                     num_replicas=num_nodes,
                                     rank=rank,
                                     )


@dataset.command
def get_base_train_set(train_groundtruth, base_dir):
    ds = MTTDataset(train_groundtruth, base_dir=base_dir)
    return ds


@dataset.command
def get_base_val_set(val_groundtruth, base_dir):
    return MTTDataset(val_groundtruth, base_dir)

@dataset.command
def get_base_test_set(test_groundtruth, base_dir):
    return MTTDatasetExhaustive(test_groundtruth, base_dir)

@dataset.command(prefix='roll_conf')
def get_roll_func(axis=-1, shift=None, shift_range=50):
    print("rolling...")

    def roll_func(b):
        x, i, y = b
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-shift_range, shift_range))
        global FirstTime

        return x.roll(sf, axis), i, y

    return roll_func


@dataset.command(prefix='norm_conf')
def get_norm_func(norm_mean=1.5880631462493773, norm_std=1.1815654825219488):
    print("normalizing...")

    def norm_func(b):
        x, i, y = b
        x = torch.as_tensor(x)

        x = (x - norm_mean) / (norm_std * 2)

        return x, i, y

    return norm_func


@dataset.command
def get_train_set(normalize, roll, wavmix=False):
    ds = get_base_train_set()
    # get_ir_sample()
    if normalize:
        ds = PreprocessDataset(ds, get_norm_func())
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())

    return ds


@dataset.command
def get_val_set(normalize):
    ds = get_base_val_set()

    if normalize:
        ds = PreprocessDataset(ds, get_norm_func())
    return ds


@dataset.command
def get_test_set(normalize):
    ds = get_base_test_set()
    if normalize:
        ds = PreprocessDataset(ds, get_norm_func())
    return ds


@dataset.command
def print_conf(_config):
    print("Config of ", dataset.path, id(dataset))
    print(_config)
    print()


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


if __name__ == "__main__":
    from sacred import Experiment

    ex = Experiment("test_dataset", ingredients=[dataset])


    @ex.automain
    def default_command():
        ex.current_run.get_command_function("print_config")()
        ds = get_test_set()
        print(ds[0])
        ds = get_train_set()
        print(ds[0])
        print("get_training_set", len(get_train_set()))
        print("get_test_set", len(get_test_set()))
