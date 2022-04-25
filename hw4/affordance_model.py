from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm, image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import imgaug as ia
# from torchinfo import summary

from common import draw_grasp


def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx] # what's this type? type RGBDataset?
        # TODO: complete this method
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
            # Where do we need to do augmentation here?
        # ===============================================================================
        rgb = data['rgb']
        center = data['center_point']

        affordace_data = {}
        # transform rgb to the correct size & range
        affordace_data['input'] = torch.permute(rgb, (2,0,1))/255
        assert affordace_data['input'].shape[0] == 3
        assert affordace_data['input'].shape[1:] == rgb.shape[:2]

        # generate target array using the get_gaussian scoremap
        affordace_data['target'] = torch.unsqueeze(torch.from_numpy(get_gaussian_scoremap(rgb.shape[:2], center.numpy())), 0)
        # print("DEBUGGING: affordance target shape is:", affordace_data['target'].shape)

        return affordace_data
        # ===============================================================================


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        Since BCEWithLogitsLoss works directly with logits, the original model does not have sigmoid layer
            Hence this extra sigmoid "layer" is needed when actually making predictions.
        """
        # print("DEBUGGING: model forwad, we have:", self.forward(x))
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here?

        # create a batch of rotated images
        rgb_rotate = []
        for i in range(8):
            aug = iaa.Affine(rotate=(22.5*i))
            rgb_rotate.append(aug(image=rgb_obs))

        # for i in range(8):
            # ia.imshow(rgb_rotate[i])

        # Some image transformation before sending to tensor and gpu
        rgb_rotate = (np.asarray(rgb_rotate)/255).transpose(0, 3, 1, 2)
        rgb_torch = torch.from_numpy(rgb_rotate).float().to(device)

        assert rgb_torch.shape == (8, 3, 128, 128)

        prediction = self.predict(rgb_torch)
        
        # find the rotation and the coordinate after rotation!
        pick = torch.argmax(prediction.squeeze()).item()
        rotation_category = pick // (128 * 128)
        y = pick % (128 * 128) // 128
        x = pick % (128 * 128) % 128

        # print("DEBUGGING: the argmax prediction picking place is:", pick)

        # Rotate the coordinate back to the original image coordinate
        
        # kps are draw on the rotated picture,
        # and then we want to rotate the picture back
        kps = KeypointsOnImage([
            Keypoint(x=x, y=y),
        ], shape=rgb_obs.shape)

        rotate_back = iaa.Sequential([
            iaa.Affine(rotate=(-22.5*rotation_category))
        ])

        rotated_rgb = iaa.Affine(rotate=(22.5*rotation_category))(image=rgb_obs)

        _, kpsaug = rotate_back(image=rotated_rgb, keypoints=kps)
        coord = (int(kpsaug.to_xy_array()[0][0]), int(kpsaug.to_xy_array()[0][1]))


        # TODO: something might be a bit off with the angle and the drawing below
        angle = rotation_category * -22.5

        # ===============================================================================
        # TODO: complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        # TODO: what does this function do???
        draw_grasp(rgb_obs, coord, angle)
        rotated_rgb = iaa.Affine(rotate=(22.5*rotation_category))(image=rgb_obs)

        output_np = prediction[rotation_category].detach().to('cpu').numpy()
        # print("DEBUGGING: output_np is", output_np)
        # output_rgb = np.stack((output_np, output_np, output_np)).transpose(1, 2, 0)


        # print("DEBUGGING: output_rgb.shape, rgb_obs.shape", output_rgb.shape, rgb_obs.shape)
        # assert output_rgb.shape == rgb_obs.shape
        # TODO: stack all 8 together in the format of their figure, and do the last line gray thing
        vis_img = []
        for i in range(8):
            if i==rotation_category:
                vis_img.append(self.visualize(rotated_rgb.transpose(2, 0, 1)/255, prediction[i].detach().to('cpu').numpy()))
            else:
                vis_img.append(self.visualize(rgb_rotate[i], prediction[i].detach().to('cpu').numpy()))
        # vis_img[-1:] = [127] * vis_img.shape[1] * vis_img.shape[2]
        vis_img = np.vstack(vis_img)
        # print("DEBUGGING: vis_img shape", vis_img.shape)

        # ===============================================================================
        return coord, angle, vis_img

