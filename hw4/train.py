from turtle import left
from typing import Dict, List, Tuple

import os
import argparse
from functools import lru_cache
from random import seed
import json
from matplotlib import image

import numpy as np
from skimage.io import imsave
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from image import read_rgb
import affordance_model
import action_regression_model
from common import save_chkpt, load_chkpt

@lru_cache(maxsize=128)
def read_rgb_cached(file_path):
    return read_rgb(file_path)

class RGBDataset(Dataset):
    def __init__(self, labels_dir: str):
        super().__init__()
        labels_path = os.path.join(labels_dir, 'labels.json')
        labels = json.load(open(labels_path, 'r'))
        label_pairs = list()
        for key, value in labels.items():
            for i, label in enumerate(value):
                label_pairs.append((
                    '{}_{}'.format(key, i),
                    label
                ))
        self.labels_dir = labels_dir
        self.label_pairs = label_pairs
    
    def __len__(self) -> int:
        return len(self.label_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Read dataset labels.
        return: 
        {
            'rgb': np.ndarray (H,W,3), torch.uint8, range [0,255]
            'center_point': np.ndarray (2,), np.float32, range [0,127]
            'angle': np.ndarray (,), np.float32, range [0, 180]
        }
        """
        key, label = self.label_pairs[idx]
        img_path = os.path.join(self.labels_dir, '{}_rgb.png'.format(key))
        rgb = read_rgb_cached(img_path)
        data = {
            'rgb': rgb,
            'center_point': np.array(label[:2], dtype=np.float32),
            'angle': np.array(label[2], dtype=np.float32)
        }
        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = torch.from_numpy(value)
        return data_torch


def get_finger_points(
        center_point: np.ndarray, 
        angle: float, width: int=10
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the pick position and angle in pixel space,
    return the position of left and right fingers of the gripper
    given the gripper width.
    """
    center_coord = np.array(center_point, dtype=np.float32)
    rad = angle / 180 * np.pi
    direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
    left_coord = center_coord - direction * width
    right_coord = center_coord + direction * width
    return left_coord, right_coord


def get_center_angle(
        left_coord: np.ndarray, 
        right_coord: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the pixel coordinate of left and right fingers to 
    gripper center and angle.
    return:
    center_coord: np.ndarray([x,y], dtype=np.float32)
    angle: float
    """
    # TODO: complete this function
    # Why do we need this function?
    # Hint: read get_finger_points
    # Hint: it's a hack
    # TODO: QUESTION: left-right right - left??
    # ===============================================================================
    center_coord = (left_coord + right_coord) / 2
    angle = np.arctan2([(right_coord-left_coord)[1]], [(right_coord-left_coord)[0]]) * 180 / np.pi
    # print("DEBUGGING: center_coord =", center_coord, "angle =", angle)
    # ===============================================================================
    return center_coord, angle


class AugmentedDataset(Dataset):
    def __init__(self, rgb_dataset: RGBDataset):
        super().__init__()
        angle_delta = 180/8
        self.rgb_dataset = rgb_dataset
        self.aug_pipeline = iaa.Sequential([
            iaa.Sometimes(
                0.7,
                iaa.Affine(
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-angle_delta/2,angle_delta/2),
                )
            ),
            
            # Improvements 3: Domain randomization
            
            # add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.Multiply((0.8, 1.2), per_channel=0.2), 
            
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

            
        ])
            
    
    def __len__(self) -> int:
        return len(self.rgb_dataset)
    
    def __getitem__(self, idx: int
            ) -> Dict[str, torch.Tensor]:
        """
        The output format should be exactly the same as RGBDataset.__getitem__
        """
        data_torch = self.rgb_dataset[idx]
        # TODO: complete this method 
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html 
        # Hint: use get_finger_points and get_center_angle
        # QUESTION: where do we do the ratation 8 times??? Is it not here?
        # ===============================================================================
        rgb = data_torch['rgb'].numpy()
        left, right = get_finger_points(data_torch['center_point'].numpy(), data_torch['angle'].numpy())

        # QUESTION: which is x and which is y again?
        kps = KeypointsOnImage([
            Keypoint(x=left[0], y=left[1]),
            Keypoint(x=right[0], y=right[1]),
        ], shape = rgb.shape)
        
        rgb_aug, kps_aug = self.aug_pipeline(image=rgb, keypoints=kps)
        left_aug = np.array(kps_aug.to_xy_array()[0]) 
        right_aug = np.array(kps_aug.to_xy_array()[1]) 

        # print("DEBUGGING: kps_aug", kps_aug.to_xy_array())
        # print("DEBUGGING: left aug is:", left_aug, "right aug is", right_aug)

        center_aug, angle_aug = get_center_angle(left_aug, right_aug)

        # transform the image input
        data_torch['rgb'] = torch.from_numpy(rgb_aug)

        # transform the keypoints
        data_torch['center_point'] = torch.from_numpy(center_aug)
        data_torch['angle'] = torch.from_numpy(angle_aug)

        # print("DEBUGGING: shape of rbg image is:", data_torch['rgb'].shape)
        # ===============================================================================
        return data_torch


def train(model, train_loader, criterion, optimizer, epoch, device):
    """
        Loop over each sample in the dataloader. Do forward + backward + optimize procedure and print mean IoU on train set.
        :param model (torch.nn.module object): miniUNet model object
        :param train_loader (torch.utils.data.DataLoader object): train dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :param epoch (int): current epoch number
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train()
    epoch_loss = []
    for cur_step, sample_batched in enumerate(train_loader):
        data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
        optimizer.zero_grad()

        # Forward pass
        # print("DEBUGGING: the trianing input data looks like:", data)
        output = model(data)
        # print("DEBUGGING: the trianing out data looks like:", output)

        loss = criterion(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Stats
        epoch_loss.append(loss.item())
        # Output stats every [steps] iteration
        if cur_step % 50 == 0:
            global_step = (epoch - 1) * len(train_loader) + cur_step
            print('step#', global_step, 'training loss', np.asarray(epoch_loss).mean())

    return np.asarray(epoch_loss).mean()


def test(model, test_loader, criterion, device, save_dir=None):
    """
        Similar to train(), but no need to backward and optimize.
        :param model (torch.nn.module object): miniUNet model object
        :param test_loader (torch.utils.data.DataLoader object): test dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.eval()
    with torch.no_grad():
        epoch_loss = []
        for i, sample_batched in enumerate(test_loader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Stats
            epoch_loss.append(loss.item())

    return np.asarray(epoch_loss).mean()


def save_prediction(
        model: torch.nn.Module, 
        dataloader: DataLoader, 
        dump_dir: str, 
        BATCH_SIZE:int
    ) -> None:
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            input = sample_batched['input'].numpy()
            target = sample_batched['target'].numpy()

            data = sample_batched['input'].to(model.device)
            # print("DEBUGGING: the input data looks like:", data)
            output = model.predict(data)
            pred = output.detach().to('cpu').numpy()

            for i in range(len(output)):
                vis_img = model.visualize(
                    input=input[i], output=pred[i], target=target[i])
                idx = batch_ID * BATCH_SIZE + i
                fname = os.path.join(dump_dir, '{:03d}.png'.format(idx))
                imsave(fname, vis_img)


def main():
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('-m', '--model', default='affordance',
        help='which model to train: "affordance" or "action_regression"')
    parser.add_argument('-a', '--augmentation', action='store_true',
        help='flag to enable data augmentation')
    parser.add_argument('-i', '--improve', default=False, action='store_true',
        help='whether you are at improving stage')
    args = parser.parse_args()

    if args.model == 'affordance':
        model_class = affordance_model.AffordanceModel
        dataset_class = affordance_model.AffordanceDataset
        max_epochs = 101
        if args.improve == False:
            model_dir = 'data/affordance'
        else:
            model_dir = 'data/affordance_improved'
    else:
        model_class = action_regression_model.ActionRegressionModel
        dataset_class = action_regression_model.ActionRegressionDataset
        max_epochs = 201
        model_dir = 'data/action_regression'
    chkpt_path = os.path.join(model_dir, 'best.ckpt')
    dump_dir = os.path.join(model_dir, 'training_vis')
    augmented_dir = os.path.join(model_dir, 'augmented_training_vis')

    seed(0)
    torch.manual_seed(0)
    ia.seed(0)

    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.improve == True:
        dataset_dir = './data/labels'
    else:
        dataset_dir = './data/labels_improve'
    raw_dataset = RGBDataset(dataset_dir)
    train_raw_dataset, test_raw_dataset = random_split(
        raw_dataset, [int(0.9 * len(raw_dataset)), len(raw_dataset) - int(0.9 * len(raw_dataset))])
    if args.augmentation:
        train_raw_dataset = AugmentedDataset(train_raw_dataset)
    train_dataset = dataset_class(train_raw_dataset)
    test_dataset = dataset_class(test_raw_dataset)
    print(f"Train dataset: {len(train_dataset)}; Test dataset: {len(test_dataset)}")

    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading model")
    model = model_class(pretrained=True)
    model.to(device)

    criterion = model.get_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    epoch = 1
    best_loss = float('inf')
    while epoch <= max_epochs:
        print('Start epoch', epoch)
        train_loss = train(model, train_loader, criterion, optimizer, epoch, device)
        test_loss = test(model, test_loader, criterion, device)
        lr_scheduler.step(test_loss)
        print('Epoch (', epoch, '/', max_epochs, ')')
        print('---------------------------------')
        print('Train loss: %0.4f' % (train_loss))
        print('Test loss: %0.4f' % (test_loss))
        print('---------------------------------')
        # Save checkpoint if is best
        if epoch % 5 == 0 and test_loss < best_loss:
            best_loss = test_loss
            save_chkpt(model, epoch, test_loss, chkpt_path)
            save_prediction(model, test_loader, dump_dir, BATCH_SIZE)
            save_prediction(model, train_loader, augmented_dir, BATCH_SIZE)
        epoch += 1

if __name__ == "__main__":
    main()
