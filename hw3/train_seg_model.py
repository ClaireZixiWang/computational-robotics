import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed
from sim import get_tableau_palette
import matplotlib.pyplot as plt


# ==================================================
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
# ==================================================


class RGBDataset(Dataset):
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # Borrowed from hw2
        # ===============================================================================
        # pass
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = img_dir
        # TODO: what to deal with the has_gt flag?
        # self.has_gt = has_gt

        # Transform to be applied on a sample.
        #  For this homework, compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean_rgb, std_rgb)])  # QUESTION: Inplace=?

        # go into the rgb/ subfolder, since all test, train, val has rgb/ subfolder
        rgb_subfolder = os.path.join(self.dataset_dir, 'rgb/')

        # borrowd from stackoverflow:
        # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
        # count # of files in a folder, excluding folders such as './' and '../'
        self.dataset_length = len([name for name in os.listdir(
            rgb_subfolder) if os.path.isfile(os.path.join(rgb_subfolder, name))])
        print("DEBUGGING: The length of the dataset is: ", self.dataset_length)
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # Borrowed from hw2
        # ===============================================================================
        return self.dataset_length
        # pass
        # # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Borrowed from hw2
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
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

        # QUESTION: should i use this? ==> YES
        gt_mask = torch.LongTensor(image.read_mask(gt_file_path))
        sample = {'input': rgb_img, 'target': gt_mask}

        # print("DEBUGGING: the dim of input image is", rgb_img.size())
        # print("DEBUGGING: the dim of output mask is", gt_mask.size())

        return sample
        # ===============================================================================


class miniUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(miniUNet, self).__init__()
        # TODO: complete this method
        # ===============================================================================
        # Some inspiration from https://github.com/milesial/Pytorch-UNet
        self.down = nn.Sequential(
            # QUESTION: any padding, stripe?
            nn.Conv2d(in_channels=n_channels, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                      padding=1),  # QUESTION: any padding, stripe?
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      padding=1),  # QUESTION: any padding, stripe?
            nn.ReLU()
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      padding=1),  # QUESTION: any padding, stripe?
            nn.ReLU()
        )
        self.down_last = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Sequential(
            # QUESTION: or should we be using convtranspose2d? --> NO
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            # QUESTION: what's the interpolate function? Upsampling?? --> Yes
            nn.Upsample(scale_factor=2)
            # QUESTION: should I concat in the forward function? --> Yes
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels=128+256, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels=64+128, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(in_channels=32+64, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(in_channels=16+32, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=n_classes, kernel_size=1)
        )
        # ===============================================================================

    def forward(self, x):
        # TODO: complete this method
        # ===============================================================================
        down1 = self.down(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down_last = self.down_last(down4)
        up1 = self.up(down_last)
        # print("The size of down1 is: ", down1.size())
        # print("The size of down2 is: ", down2.size())
        # print("The size of down3 is: ", down3.size())
        # print("The size of down4 is: ", down4.size())
        # print("The size of down_last is: ", down_last.size())
        # QUESTION: what should be the axis?
        up2 = self.up2(torch.cat([down4, up1], axis=1))
        up3 = self.up3(torch.cat([down3, up2], axis=1))
        up4 = self.up4(torch.cat([down2, up3], axis=1))
        output = self.up5(torch.cat([down1, up4], axis=1))
        # print("The size of up1 is: ", up1.size())
        # print("The size of up2 is: ", up2.size())
        # print("The size of up3 is: ", up3.size())
        # print("The size of up4 is: ", up4.size())
        # print("The size of output is: ", output.size())
        # print("The output is:", output)
        return output
        # ===============================================================================


def save_chkpt(model, epoch, test_miou, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': test_miou, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
        Load model parameters from saved checkpoint.
        :param model (torch.nn.module object): miniUNet model to accept the saved parameters.
        :param chkpt_path (str): path of the checkpoint to be loaded.
        :return model (torch.nn.module object): miniUNet model with its parameters loaded from the checkpoint.
        :return epoch (int): epoch at which the checkpoint is saved.
        :return model_miou (float): miou of the test set at the checkpoint.
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_prediction(model, dataloader, dump_dir, device, BATCH_SIZE):
    """
        For all datapoints d in dataloader, save  ground truth segmentation mask (as {id}.png)
          and predicted segmentation mask (as {id}_pred.png) in dump_dir.
        :param model (torch.nn.module object): trained miniUNet model
        :param dataloader (torch.utils.data.DataLoader object): dataloader to use for getting predictions
        :param dump_dir (str): dir path for dumping predictions
        :param device (torch.device object): pytorch cpu/gpu device object
        :param BATCH_SIZE (int): batch size of dataloader
        :return: None
    """
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            data, target = sample_batched['input'].to(
                device), sample_batched['target'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                gt_image = convert_seg_split_into_color_image(
                    target[i].cpu().numpy())
                pred_image = convert_seg_split_into_color_image(
                    pred[i].cpu().numpy())
                combined_image = np.concatenate((gt_image, pred_image), axis=1)
                test_ID = batch_ID * BATCH_SIZE + i
                image.write_mask(
                    combined_image, f"{dump_dir}/{test_ID}_gt_pred.png")


def save_learning_curve(train_loss_list, train_miou_list, test_loss_list, test_miou_list):
    """
    In:
        train_loss, train_miou, test_loss, test_miou: list of floats, where the length is how many epochs you trained.
    Out:
        None.
    Purpose:
        Plot and save the learning curve.
    """
    epochs = np.arange(1, len(train_loss_list)+1)
    plt.figure()
    lr_curve_plot = plt.plot(epochs, train_loss_list,
                             color='navy', label="train_loss")
    plt.plot(epochs, train_miou_list, color='teal', label="train_mIoU")
    plt.plot(epochs, test_loss_list, color='orange', label="test_loss")
    plt.plot(epochs, test_miou_list, color='gold', label="val_mIoU")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(epochs, epochs)
    plt.yticks(np.arange(10)*0.1, [f"0.{i}" for i in range(10)])
    plt.xlabel('epoch')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.savefig('learning_curve.png', bbox_inches='tight')
    plt.show()


def iou(pred, target, n_classes=4):
    """
        Compute IoU on each object class and return as a list.
        :param pred (np.array object): predicted mask
        :param target (np.array object): ground truth mask
        :param n_classes (int): number of classes
        :return cls_ious (list()): a list of IoU on each object class
    """
    cls_ious = []
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(1, n_classes):  # class 0 is background
        pred_P = pred == cls
        target_P = target == cls
        pred_N = ~pred_P
        target_N = ~target_P
        if target_P.sum() == 0:
            # print("class", cls, "doesn't exist in target")
            continue
        else:
            intersection = pred_P[target_P].sum()  # TP
            if intersection == 0:
                # print("pred and target for class", cls, "have no intersection")
                cls_ious.append(float(0))
                continue
            else:
                FP = pred_P[target_N].sum()
                FN = pred_N[target_P].sum()
                union = intersection + FN + FP  # or pred_P.sum() + target_P.sum() - intersection
                cls_ious.append(float(intersection) / float(union))
    # print("DEBUGGING:", cls_ious)
    return cls_ious


def run(model, device, loader, criterion, is_train=False, optimizer=None):
    """
        Run forward pass for each sample in the dataloader. Run backward pass and optimize if training.
        Calculate and return mean_epoch_loss and mean_iou
        :param model (torch.nn.module object): miniUNet model object
        :param loader (torch.utils.data.DataLoader object): dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param is_train (bool): True if training
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train(is_train)
    # TODO: complete this function
    # ===============================================================================
    mean_epoch_loss = 0
    mean_iou = 0
    for i, data in enumerate(loader):
        inputs = data['input']
        ground_truth = data['target']
        # print("DEBUGGING: printing the ground_truth", ground_truth)
        # print("DEBUGGING: rgb size are:", inputs.size(), "and the gt size are:", ground_truth.size())
        inputs = inputs.to(device)

        # print(type(inputs))

        # outputs = model(inputs)
        # print('output shape:', outputs.shape)

        # labels = torch.zeros(inputs.shape)
#         for c in range(inputs.shape[1]):
#             class_mask = torch.full(ground_truth.shape, c)
#             labels[c] = (class_mask == ground_truth).long()

#         print(labels)
        ground_truth = ground_truth.to(device)
        # labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()  # QUESTION: What does this do??

        # forward + backword step
        outputs = model(inputs)
        # print("DEBUGGING: the dimension of the model output is", outputs.size())
        # QUESTION: can I apply cross entropy loss directly to non-one hotted masks?
        loss = criterion(outputs, ground_truth)
        _, pred = torch.max(outputs, dim=1)
        # print("DEBUGGING: the dimension of the predicted mask is", pred.size())

        if is_train:
            loss.backward()
            optimizer.step()

        # reporting statistics
        mean_epoch_loss += loss.item()

        # print("DEBUGGING: I'm training \[",is_train,"\], and the batch size is", len(inputs))

        # in every batch, calculate the iou for each datapoint, then sum them up and calculate the batch miou
        batch_iou = 0
        for i in range(len(inputs)):
            miou_data_point = np.array(iou(pred[i], ground_truth[i])).mean()
            batch_iou += miou_data_point
        mean_iou += batch_iou / len(inputs)

    mean_epoch_loss /= len(loader)
    mean_iou /= len(loader)

    if is_train:
        print('The training loss for the epoch is %f and the training iou is %f' % (
            mean_epoch_loss, mean_iou))
    else:
        print('The validation loss for the epoch is %f and the training iou is %f' % (
            mean_epoch_loss, mean_iou))

    return mean_epoch_loss, mean_iou
    # ===============================================================================


def convert_seg_split_into_color_image(img):
    color_palette = get_tableau_palette()
    colored_mask = np.zeros((*img.shape, 3))

    print(np.unique(img))

    for i, unique_val in enumerate(np.unique(img)):
        if unique_val == 0:
            obj_color = np.array([0, 0, 0])
        else:
            obj_color = np.array(color_palette[i-1]) * 255
        obj_pixel_indices = (img == unique_val)
        colored_mask[:, :, 0][obj_pixel_indices] = obj_color[0]
        colored_mask[:, :, 1][obj_pixel_indices] = obj_color[1]
        colored_mask[:, :, 2][obj_pixel_indices] = obj_color[2]
    return colored_mask.astype(np.uint8)


if __name__ == "__main__":
    # ==============Part 4 (a) Training Segmentation model ================
    # Complete all the TODO's in this file
    # - HINT: Most TODO's in this file are exactly the same as homework 2.

    seed(0)
    torch.manual_seed(0)

    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Define directories
    root_dir = './dataset/'
    train_dir = root_dir + 'train/'
    val_dir = root_dir + 'val/'
    test_dir = root_dir + 'test/'

    # TODO: Prepare train and test datasets
    # Load the "dataset" directory using RGBDataset class as a pytorch dataset
    # Split the above dataset into train and test dataset in 9:1 ratio using `torch.utils.data.random_split` method
    # ===============================================================================
    dataset = RGBDataset(root_dir)
    train_size = int(0.9 * len(dataset))
    test_size = int(0.1 * len(dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    # ===============================================================================

    # TODO: Prepare train and test Dataloaders. Use appropriate batch size
    # ===============================================================================
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    # ===============================================================================

    # TODO: Prepare model
    # ===============================================================================
    model = miniUNet(n_channels=3, n_classes=4)
    model.to(device)  # QUESTION: should I do this here or in the training loop?
    # ===============================================================================

    # TODO: Define criterion, optimizer and learning rate scheduler
    # ===============================================================================
    criterion = torch.nn.CrossEntropyLoss()
    # QUESTION: what hyperparam for Adam?
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-8)
    # ===============================================================================

    # TODO: Train and test the model.
    # Tips:
    # - Remember to save your model with best mIoU on objects using save_chkpt function
    # - Try to achieve Test mIoU >= 0.9 (Note: the value of 0.9 only makes sense if you have sufficiently large test set)
    # - Visualize the performance of a trained model using save_prediction method. Make sure that the predicted segmentation mask is almost correct.
    # ===============================================================================
    train_loss_list, train_miou_list, test_loss_list, test_miou_list = list(
    ), list(), list(), list()
    epoch, max_epochs = 0, 5  # TODO: you may want to make changes here
    best_miou = float('-inf')
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        train_loss, train_miou = run(
            model, device, train_loader, criterion, True, optimizer)
        test_loss, test_miou = run(model, device, test_loader, criterion)
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
        test_loss_list.append(test_loss)
        test_miou_list.append(test_miou)
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
        print('Testing loss & mIoU: %0.2f %0.2f' % (test_loss, test_miou))
        print('---------------------------------')
        if test_miou > best_miou:
            best_miou = test_miou
            save_chkpt(model, epoch, test_miou, 'checkpoint_multi.pth.tar')
        epoch += 1

    # Load the best checkpoint, use save_prediction() on the validation set and test set
    model, epoch, best_miou = load_chkpt(
        model, 'checkpoint_multi.pth.tar', device)
    save_prediction(model, test_loader, test_dir, device, 4)
    save_learning_curve(train_loss_list, train_miou_list,
                        test_loss_list, test_miou_list)

    # ===============================================================================
