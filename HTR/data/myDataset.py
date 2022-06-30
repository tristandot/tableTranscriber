# Imports for myDataset
import data.dataset_loader
from data.Preprocessing import preprocessing, pad_packed_collate
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class myDataset(Dataset):
    def __init__(self, set='train', data_size=(32, None),
                 affine=False, centered=False, deslant=False, data_aug=False, keep_ratio=True, enhance_contrast=False):
        self.data_size  = data_size
        self.affine     = affine
        self.centered   = centered
        self.deslant    = deslant
        self.keep_ratio = keep_ratio
        self.enhance_contrast = enhance_contrast
        self.data_aug   = data_aug
        self.data = data.dataset_loader.main_loader(set)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=(-2, 2), translate=(0, 0), scale=(0.9, 1),
                                        shear=5, resample=False, fillcolor=255),
                # Other possible data augmentation
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=255)
            ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item][0]
        gt  = self.data[item][1]
        # data pre-processing
        img = preprocessing(img, self.data_size, affine=self.affine,
                            centered=self.centered, deslant=self.deslant, keep_ratio=self.keep_ratio,
                            enhance_contrast=self.enhance_contrast)
        # data augmentation
        if self.data_aug:
            img = torch.Tensor(img).float().unsqueeze(0)
            img = self.transform(img)
            img = transforms.ToTensor()(img).float()
        else:
            img = torch.Tensor(img).float().unsqueeze(0)

        return img, gt

# test above functions
if __name__ == '__main__':
    test_set = myDataset(data_size=(32, 400), set='test', data_aug=True, keep_ratio=True)
    print("len(test_et_set) =", test_set.__len__())
    # augmentation using data sampler
    batch_size = 8
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=pad_packed_collate)
    for iter_idx, (img, gt) in enumerate(test_loader):
        print("img.size() =", img.data.size())
        print("gt =", gt)
        if iter_idx == 2:
            break