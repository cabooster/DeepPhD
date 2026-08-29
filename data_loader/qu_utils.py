"""Dataset partitioning helpers: crop coordinates, load stacks, stitch test patches."""
import numpy as np
import os
import tifffile as tiff
import math
from data_loader.expand_temporal_dimension import expand_temporal_dimension

class prepare_dataset():
    """Build overlapping 3D patch indices for multi-stack training volumes."""

    def __init__(self, settings):
        self.overlap_factor = settings.overlap_factor
        self.patch_t = settings.patch_t
        self.patch_x = settings.patch_x
        self.patch_y = settings.patch_y
        self.select_img_num = settings.select_img_num
        self.train_datasets_size = settings.train_datasets_size
        self.datasets_path = settings.datasets_path
        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))  # patch gap in x
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))  # patch gap in y
        

    def get_gap_t(self):
        """Compute temporal stride ``gap_t`` and number of temporal samples ``s_num``."""
        w_num = math.ceil((self.whole_x - self.patch_x) / self.gap_x) + 1
        h_num = math.ceil((self.whole_y - self.patch_y) / self.gap_y) + 1
        s_num = math.ceil(self.train_datasets_size / w_num / h_num / self.stack_num)

        if s_num <= 1: s_num = 2
        gap_t = math.floor((self.whole_t-self.patch_t*2)/(s_num-1))
        if gap_t < 1: gap_t = 1
        return gap_t, s_num

    def train_preprocess_lessMemoryMulStacks(self):
        """
        The original noisy stack is partitioned into thousands of 3D sub-stacks (patch) with the setting
        overlap factor in each dimension.

        Important Fields:
           self.name_list : the coordinates of 3D patch are indexed by the patch name in name_list.
           self.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack.
           self.stack_index : the index of the noisy stacks.
           self.noise_im_all : the collection of all noisy stacks.

        """
        self.name_list = []
        self.coordinate_list = {}
        self.stack_index = []
        self.noise_im_all = []
        ind = 0

        if self.datasets_path[-1]!='/':
           self.datasets_name=self.datasets_path.split("/")[-1]
        else:
            self.datasets_name=self.datasets_path.split("/")[-2]
        
        self.stack_num = len(list(os.walk(self.datasets_path, topdown=False))[-1][-1])
        print('\033[1;31mImage list for training -----> \033[0m')
        print('Total stack number -----> ', self.stack_num)

        for im_name in list(os.walk(self.datasets_path, topdown=False))[-1][-1]:
            print('Noise image name -----> ', im_name)
            im_dir = self.datasets_path + '//' + im_name
            noise_im = tiff.imread(im_dir)
            if noise_im.shape[0] > self.select_img_num:
                noise_im = noise_im[0:self.select_img_num, :, :]

            if noise_im.shape[0] < 400:
                noise_im = expand_temporal_dimension(noise_im, target_frames=400)

            print(noise_im.shape)
            self.whole_x = noise_im.shape[2]
            self.whole_y = noise_im.shape[1]
            self.whole_t = noise_im.shape[0]
            print('Noise image shape -----> ', noise_im.shape)
            # Calculate real gap_t
            self.gap_t, s_num = self.get_gap_t()
            noise_im = noise_im.astype(np.float32)

            self.noise_im_all.append(noise_im)

            
            patch_t2 = self.patch_t * 2
            cut_w = (self.patch_x - self.gap_x) / 2
            cut_h = (self.patch_y - self.gap_y) / 2
            num_w = math.ceil((self.whole_x - self.patch_x + self.gap_x) / self.gap_x)
            num_h = math.ceil((self.whole_y - self.patch_y + self.gap_y) / self.gap_y)
            delta_x = self.whole_x - (self.patch_x + (num_w - 2) * self.gap_x)
            delta_y = self.whole_y - (self.patch_y + (num_h - 2) * self.gap_y)
            for x in range(0, math.ceil((self.whole_x - self.patch_x + self.gap_x) / self.gap_x)):
                for y in range(0, math.ceil((self.whole_y - self.patch_y + self.gap_y) / self.gap_y)):
                    for z in range(0, s_num):
                        single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0, 'patch_start_w':0, 'patch_end_w':0, 'patch_start_h':0, 'patch_end_h':0}
                        if y != (num_h - 1):
                            init_h = self.gap_y * y
                            end_h = self.gap_y * y + self.patch_y
                        elif y == (num_h - 1):
                            init_h = self.whole_y - self.patch_y
                            end_h = self.whole_y

                        if x != (num_w - 1):
                            init_w = self.gap_x * x
                            end_w = self.gap_x * x + self.patch_x
                        elif x == (num_w - 1):
                            init_w = self.whole_x - self.patch_x
                            end_w = self.whole_x

                        init_s = self.gap_t * z
                        end_s = self.gap_t * z + patch_t2
                        single_coordinate['init_h'] = init_h
                        single_coordinate['end_h'] = end_h
                        single_coordinate['init_w'] = init_w
                        single_coordinate['end_w'] = end_w
                        single_coordinate['init_s'] = init_s
                        single_coordinate['end_s'] = end_s
                        
                        if x == 0:
                            single_coordinate['patch_start_w'] = 0
                            single_coordinate['patch_end_w'] = self.patch_x - cut_w
                            if x == num_w - 2:
                                single_coordinate['patch_end_w'] = self.patch_x - (self.patch_x - delta_x) / 2
                            if x == num_w - 1:
                                single_coordinate['patch_end_w'] = self.patch_x
                        elif x == num_w - 2:
                            single_coordinate['patch_start_w'] = cut_w
                            single_coordinate['patch_end_w'] = self.patch_x - (self.patch_x - delta_x) / 2
                        elif x == num_w - 1:
                            single_coordinate['patch_start_w'] = (self.patch_x - delta_x) / 2
                            single_coordinate['patch_end_w'] = self.patch_x
                        else:
                            single_coordinate['patch_start_w'] = cut_w
                            single_coordinate['patch_end_w'] = self.patch_x - cut_w

                        if y == 0:
                            single_coordinate['patch_start_h'] = 0
                            single_coordinate['patch_end_h'] = self.patch_y - cut_h
                            if y == num_h - 2:
                                single_coordinate['patch_end_h'] = self.patch_y - (self.patch_y - delta_y) / 2
                            if y == num_h - 1:
                                single_coordinate['patch_end_h'] = self.patch_y
                        elif y == num_h - 2:
                            single_coordinate['patch_start_h'] = cut_h
                            single_coordinate['patch_end_h'] = self.patch_y - (self.patch_y - delta_y) / 2
                        elif y == num_h - 1:
                            single_coordinate['patch_start_h'] = (self.patch_y - delta_y) / 2
                            single_coordinate['patch_end_h'] = self.patch_y
                        else:
                            single_coordinate['patch_start_h'] = cut_h
                            single_coordinate['patch_end_h'] = self.patch_y - cut_h

                        patch_name = self.datasets_name + '_' + im_name.replace('.tif', '') + '_x' + str(
                            x) + '_y' + str(y) + '_z' + str(z)
                        self.name_list.append(patch_name)
                        self.coordinate_list[patch_name] = single_coordinate
                        self.stack_index.append(ind)

            ind = ind + 1
        return self.name_list, self.coordinate_list, self.noise_im_all, self.stack_index, noise_im.shape, num_w
    
def test_preprocess_chooseOne_real(args, img_id):
    """
    Choose one original noisy stack and partition it into thousands of 3D sub-stacks (patch) with the setting
    overlap factor in each dimension.

    Args:
        args : the train object containing input params for partition
        img_id : the id of the test image
    Returns:
        name_list : the coordinates of 3D patch are indexed by the patch name in name_list
        noise_im : the original noisy stacks
        coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack
        im_name : the file name of the noisy stacks

    """
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t2 = int(args.patch_t * (1 - args.overlap_factor))
    if gap_t2 == 0:
        gap_t2 = 1
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t2 - gap_t2) / 2
    im_folder = args.datasets_path

    name_list = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    im_name = img_list[img_id]

    if args.datasets_path[-1]!='/':
        datasets_name=args.datasets_path.split("/")[-1]
    else:
        datasets_name=args.datasets_path.split("/")[-2]


    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)
    input_data_type = noise_im.dtype

    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[0:args.test_datasize, :, :]
    original_T = noise_im.shape[0]
    if noise_im.shape[0] < 400:
        noise_im = expand_temporal_dimension(noise_im, target_frames=400)
    if img_id == 0:
        print('Testing image name -----> ', im_name)

    noise_im = noise_im.astype(np.float32)

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)
    for z in range(0, num_s):
        for x in range(0, num_h):
            for y in range(0, num_w):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s - 1):
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if num_w == 1:
                    single_coordinate['stack_start_w'] = 0
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x
                elif y == 0:
                    single_coordinate['stack_start_w'] = y * gap_x
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x - cut_w
                elif y == num_w - 1:
                    single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y * gap_x + cut_w
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x - cut_w

                if num_h == 1:
                    single_coordinate['stack_start_h'] = 0
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y
                elif x == 0:
                    single_coordinate['stack_start_h'] = x * gap_y
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y - cut_h
                elif x == num_h - 1:
                    single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x * gap_y + cut_h
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y - cut_h

                if num_s == 1:
                    single_coordinate['stack_start_s'] = 0
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2
                elif z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t2
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s

                patch_name = datasets_name + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                name_list.append(patch_name)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list, im_name, input_data_type, num_w, original_T

def singlebatch_test_save(single_coordinate, output_image):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks (if the batch size equal to 1).

    Args:
        single_coordinate : the coordinate dict of the image
        output_image : the output sub-stack of the network
    Returns:
        output_patch : the output patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    output_patch = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save(single_coordinate, id, output_image):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks. (if the batch size larger than 1).

    Args:
        single_coordinate : the coordinate dict of the image
        output_image : the output sub-stack of the network
    Returns:
        output_patch : the output patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate['stack_start_s'].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate['stack_end_s'].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate['patch_end_s'].numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_image_id = output_image[id]
    output_patch = output_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s
