import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO
        # Some inspiration from https://github.com/milesial/Pytorch-UNet
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      padding=1),  # QUESTION: any padding, stripe?
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
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=1)
        )

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO
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


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
