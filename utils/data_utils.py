import os
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))
        if mask is not None:
            mask = transform.resize(mask, (new_h, new_w))
        
        return {'image': image, 'mask': mask }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask) if mask is not None else None}
                

composed = transforms.Compose([Rescale(128), ToTensor()])

class TgsDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir=None, transform=composed):
        self.image_df = df
        self.img_dir = img_dir
        self.transform = transform
        self.mask_dir = mask_dir
        
    def __len__(self):
        return len(self.image_df)
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.img_dir,self.image_df.iloc[idx, 0] + '.png')
        image = io.imread(img_path)
        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir,self.image_df.iloc[idx, 0] + '.png')
            mask = io.imread(mask_path)
        sample = {'image':image, 'mask':mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample if sample['mask'] is not None else {'image':sample['image']}
        
