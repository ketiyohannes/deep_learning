import numpy as np
import torch
from sklearn.datasets import load_sample_images


sample_images = np.stack(load_sample_images()["images"])
print("before", sample_images.shape)
sample_images = torch.tensor(sample_images, dtype=torch.float32) / 255

print("after zeroing", sample_images.shape)


# permuting the channels cause pytorch expects color channel dimension before the height and width

sample_images_permuted = sample_images.permute(0, 3, 1, 2)
print("after permuting", sample_images_permuted.shape)

import torchvision
import torchvision.transforms.v2 as T

cropped_images = T.CenterCrop((70, 120))(sample_images_permuted)
print("after cropping", cropped_images.shape)



import torch.nn as nn

torch.manual_seed(42)
conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding="same")
fmaps = conv_layer(cropped_images)
print("fmaps shape", fmaps.shape)


