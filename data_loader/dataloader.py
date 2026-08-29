from torch.utils.data import Dataset
from torch.utils.data import Sampler
import torch
import random
import numpy as np

def random_transform(y_a, y_b):
    """
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        y_a, y_b: interlaced raw branches before data augmentation
    Return:
        y_a, y_b: interlaced raw branches after data augmentation
    """
    augmentation_transform = random.randrange(8)
    if augmentation_transform == 1:  # left rotate 90
        y_a = np.rot90(y_a, k=1, axes=(1, 2))
        y_b = np.rot90(y_b, k=1, axes=(1, 2))
    elif augmentation_transform == 2:  # left rotate 180
        y_a = np.rot90(y_a, k=2, axes=(1, 2))
        y_b = np.rot90(y_b, k=2, axes=(1, 2))
    elif augmentation_transform == 3:  # left rotate 270
        y_a = np.rot90(y_a, k=3, axes=(1, 2))
        y_b = np.rot90(y_b, k=3, axes=(1, 2))
    elif augmentation_transform == 4:  # horizontal flip
        y_a = y_a[:, :, ::-1]
        y_b = y_b[:, :, ::-1]
    elif augmentation_transform == 5:  # horizontal flip & left rotate 90
        y_a = y_a[:, :, ::-1]
        y_a = np.rot90(y_a, k=1, axes=(1, 2))
        y_b = y_b[:, :, ::-1]
        y_b = np.rot90(y_b, k=1, axes=(1, 2))
    elif augmentation_transform == 6:  # horizontal flip & left rotate 180
        y_a = y_a[:, :, ::-1]
        y_a = np.rot90(y_a, k=2, axes=(1, 2))
        y_b = y_b[:, :, ::-1]
        y_b = np.rot90(y_b, k=2, axes=(1, 2))
    elif augmentation_transform == 7:  # horizontal flip & left rotate 270
        y_a = y_a[:, :, ::-1]
        y_a = np.rot90(y_a, k=3, axes=(1, 2))
        y_b = y_b[:, :, ::-1]
        y_b = np.rot90(y_b, k=3, axes=(1, 2))

    return y_a, y_b, augmentation_transform

class trainset(Dataset):
    """
    Train set generator for pytorch training

    """

    def __init__(self, patch_names, patch_coordinates_by_name, raw_stacks, stack_index):
        self.patch_names = patch_names
        self.patch_coordinates_by_name = patch_coordinates_by_name
        self.raw_stacks = raw_stacks
        self.stack_index = stack_index

    def __getitem__(self, index):
        """
        For temporal stacks with a small lateral size or short recording period, patches can be
        randomly cropped from the original stack to augment the training set according to the record
        coordinate. Then, interlaced frames of each patch are extracted to form two 3D tiles.
        One of them serves as the input and the other serves as the target for network training
        Args:
            index : the index of 3D patchs used for training
        Return:
            y_a, y_b: interlaced raw branches used for bidirectional self-supervision
        """
        stack_index = self.stack_index[index]
        raw_stack = self.raw_stacks[stack_index]
        patch_coordinates = self.patch_coordinates_by_name[self.patch_names[index]]
        init_h = patch_coordinates['init_h']
        end_h = patch_coordinates['end_h']
        init_w = patch_coordinates['init_w']
        end_w = patch_coordinates['end_w']
        init_s = patch_coordinates['init_s']
        end_s = patch_coordinates['end_s']
        patch_start_w = patch_coordinates['patch_start_w']
        patch_end_w = patch_coordinates['patch_end_w']
        patch_start_h = patch_coordinates['patch_start_h']
        patch_end_h = patch_coordinates['patch_end_h']
        y_a = raw_stack[init_s:end_s:2, init_h:end_h, init_w:end_w]
        y_b = raw_stack[init_s + 1:end_s:2, init_h:end_h, init_w:end_w]

        if random.random() >= 0.5:
            y_a, y_b = y_b, y_a
        y_a, y_b, augmentation_transform = random_transform(y_a, y_b)

        y_a = torch.from_numpy(y_a.copy())
        y_b = torch.from_numpy(y_b.copy())
         
        training_sample = {'y_a' : y_a, 
                  'y_b' : y_b,
                  'init_h' : init_h,
                  'end_h' : end_h,
                  'init_w' : init_w,
                  'end_w' : end_w, 
                  'patch_start_w' : patch_start_w,
                  'patch_end_w' : patch_end_w,
                  'patch_start_h' : patch_start_h,
                  'patch_end_h' : patch_end_h,
                  'augmentation_transform' : augmentation_transform}
        return training_sample

    def __len__(self):
        return len(self.patch_names)
    
    def len(self):
        return len(self.patch_names)
    
class FixedSizeGroupBatchSampler(Sampler):
    """Batch sampler that keeps patches from the same image row together.

    Dataset indices are laid out row-wise with ``patches_per_row`` patches per row.
    Each row is split into groups of ``group_size`` for RN estimation along a full row.
    """

    def __init__(self, dataset_len, patches_per_row, group_size, shuffle_batch=True, shuffle_within_group=True):
        self.dataset_len = dataset_len
        self.patches_per_row = patches_per_row 
        self.group_size = group_size
        self.shuffle_batch = shuffle_batch
        self.shuffle_within_group = shuffle_within_group

        assert dataset_len % patches_per_row == 0, "dataset_len must be an integer multiple of patches_per_row"
        self.total_rows = dataset_len // patches_per_row

    def __iter__(self):
        batches = []

        for row_index in range(self.total_rows):
            row_start_index = row_index * self.patches_per_row
            patch_indices_in_row = list(range(row_start_index, row_start_index + self.patches_per_row))

            if self.shuffle_within_group:
                random.shuffle(patch_indices_in_row)

            # Split the row into fixed-size groups of group_size.
            for i in range(0, self.patches_per_row, self.group_size):
                batch_indices = patch_indices_in_row[i:i + self.group_size]

                batches.append(batch_indices)

        # Shuffle batch order across rows.
        if self.shuffle_batch:
            random.shuffle(batches)

        for batch_indices in batches:
            yield batch_indices

    def __len__(self):
        from math import ceil
        return self.total_rows * ceil(self.patches_per_row / self.group_size)
        
class testset(Dataset):
    """
    Test set generator for pytorch inference

    """

    def __init__(self, patch_names, patch_coordinates_by_name, raw_stack):
        self.patch_names = patch_names
        self.patch_coordinates_by_name = patch_coordinates_by_name
        self.raw_stack = raw_stack

    def __getitem__(self, index):
        """
        Generate patches of the noisy image.
        Args:
            index : the index of 3D patch used for testing
        Return:
            y_patch : the patch of the noisy image
            single_coordinate : the specific coordinate of patches in the noisy image for stitching
        """
        patch_coordinates = self.patch_coordinates_by_name[self.patch_names[index]]
        init_h = patch_coordinates['init_h']
        end_h = patch_coordinates['end_h']
        init_w = patch_coordinates['init_w']
        end_w = patch_coordinates['end_w']
        init_s = patch_coordinates['init_s']
        end_s = patch_coordinates['end_s']
        y_patch = self.raw_stack[init_s:end_s, init_h:end_h, init_w:end_w]
        y_patch = torch.from_numpy(y_patch)
        return y_patch, patch_coordinates

    def __len__(self):
        return len(self.patch_names)
