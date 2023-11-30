import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms, io

import image


class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt):
        """
        In:
            dataset_dir: string, train_dir, val_dir, and test_dir in segmentation.py.
                         Be careful the images are stored in the subfolders under these directories.

                         eg: root_dir = './dataset/'
                             train_dir = root_dir + 'train/'
                             val_dir = root_dir + 'val/'
                             test_dir = root_dir + 'test/'

            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        # TODO: transform to be applied on a sample.
        #  For this homework, compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean_rgb, std_rgb)])  # QUESTION: Inplace=?
        # TODO: number of samples in the dataset.
        #  You'd better not hard code the number,
        #  because this class is used to create train, validation and test dataset
        #  (which have different sizes).

        # go into the rgb/ subfolder, since all test, train, val has rgb/ subfolder
        rgb_subfolder = os.path.join(self.dataset_dir, 'rgb/')

        # borrowd from stackoverflow:
        # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
        # count # of files in a folder, excluding folders such as './' and '../'
        self.dataset_length = len([name for name in os.listdir(
            rgb_subfolder) if os.path.isfile(os.path.join(rgb_subfolder, name))])
        print("CHECKING: The length of the dataset is: ", self.dataset_length)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
            For every image, prepare the image so that it can be fed to dataloader
                For training images, read in image, transform(toTensor, normalize, etc)
                For targets, read in targets, cast to tensor (Normally we use target_transform(toTensor()))
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.
        rgb_subfolder = os.path.join(self.dataset_dir, 'rgb/')
        gt_subfolder = os.path.join(self.dataset_dir, 'gt/')

        rgb_name = str(idx) + '_rgb.png'
        gt_name = str(idx) + '_gt.png'

        rgb_file_path = os.path.join(rgb_subfolder, rgb_name)
        gt_file_path = os.path.join(gt_subfolder, gt_name)

        # print("CHECKING: rgb_file_path: %s; gt_file_path: %s" % (rgb_file_path, gt_file_path))

        rgb_img = self.transform(image.read_rgb(rgb_file_path))
        # QUESTION: should I use torch.io.read_image()?  ==> NO
        # QUESTION: why is the generated picture so dim?  ==> Didn't transform!!
        # QUESTION: if I use image.read_rgb, do I need to transform it to tensor? ==> YES
        # print("CHECKING: type of rgb_img is:", type(rgb_img))

        if self.has_gt is False:
            sample = {'input': rgb_img}
        else:
            # QUESTION: should i use this? ==> YES
            gt_mask = torch.LongTensor(image.read_mask(gt_file_path))
            sample = {'input': rgb_img, 'target': gt_mask}

        return sample
