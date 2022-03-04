import matplotlib.pyplot as plt
import numpy as np
from dataset import RGBDataset
import image


def denormalize_rgb(rgb):
    """
    In:
        rgb: Tensor [3, height, width].
    Out:
        rgb: Tensor [3, height, width].
    Purpose:
        Denormalize an RGB image.
    """
    mean_rgb = [0.722, 0.751, 0.807]
    std_rgb = [0.171, 0.179, 0.197]
    for i in range(rgb.shape[0]):
        rgb[i] = np.maximum(rgb[i] * std_rgb[i] + mean_rgb[i], 0)
    return rgb


def show_rgb(rgb_img):
    """
    In:
        rgb_img: Numpy array [height, width, 3].
    Out:
        None.
    Purpose:
        Visualize an RGB image.
    """
    plt.figure()
    plt.imshow(rgb_img)
    plt.show()


def put_palette(obj_id):
    """
    In:
        obj_id: int.
    Out:
        None.
    Purpose:
        Fetch the mask color of specific object.
    """
    mypalette = np.array(
        [[0, 0, 0],
         [255, 0, 0],
         [0, 255, 0],
         [0, 0, 255],
         [255, 255, 0],
         [255, 0, 255],
         ],
        dtype=np.uint8,
    )
    return mypalette[obj_id]


def mask2rgb(mask):
    """
    In:
        mask: Numpy array [height, width].
    Out:
        None.
    Purpose:
        Convert a mask to RGB image for visualization.
    """
    v_put_palette = np.vectorize(put_palette, signature='(n)->(n,3)')
    return v_put_palette(mask.flatten()).reshape(mask.shape[0], mask.shape[1], 3)


def show_mask(mask):
    """
    In:
        mask: Numpy array [height, width].
    Out:
        None.
    Purpose:
        Visualize a mask.
    """
    show_rgb(mask2rgb(mask))


def check_dataset(dataset):
    """
    In:
        dataset: RGBDataset instance in this homework.
    Out:
        None.
    Purpose:
        Test dataset by visualizing a random sample.
    """

    print("dataset size:", len(dataset))
    sample = dataset[np.random.randint(len(dataset))]
    rgb = sample['input'].numpy()
    print("input shape:", rgb.shape)
    show_rgb(denormalize_rgb(rgb).transpose(1, 2, 0))
    if dataset.has_gt is True:
        mask = sample['target'].numpy()
        print("target shape:", mask.shape)
        show_mask(mask)


def check_dataloader(dataloader):
    """
    In:
        dataloader: Dataloader instance.
    Out:
        None.
    Purpose:
        Test dataloader by visualizing a batch.
    """
    print("dataset size:", len(dataloader.dataset))
    dataiter = iter(dataloader)
    sample = dataiter.next()
    # print(sample)
    # inputs, ground_truth = sample
    # print("====================================")
    # print(inputs, ground_truth)
    rgb = sample['input'].numpy()
    print("input shape:", rgb.shape)
    if dataloader.dataset.has_gt is True:
        mask = sample['target'].numpy()
        print("target shape:", mask.shape)
    for i in range(rgb.shape[0]):
        show_rgb(denormalize_rgb(rgb[i]).transpose(1, 2, 0))
        if dataloader.dataset.has_gt is True:
            show_mask(mask[i])

# visualize a ground truth 
# TODO: Probably should not be here, might be put in a separate test.py file
def main():
    # show_mask(image.read_mask('/Users/zixiwang/dev/comsw4733-comp-robotics/hw2/dataset/train/gt/0_gt.png'))
    # check_dataset(RGBDataset('/Users/zixiwang/dev/comsw4733-comp-robotics/hw2/dataset/test', has_gt=False))
    check_dataset(RGBDataset('/Users/zixiwang/dev/comsw4733-comp-robotics/hw2/dataset/train', has_gt=True))


if __name__ == "__main__":
    main()
