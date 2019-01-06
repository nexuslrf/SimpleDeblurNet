import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_list, crop_size=(256, 256), path='./'):
        super(type(self), self).__init__()
        self.blur_list = img_list['blur'] #+ img_list['sharp']
        self.sharp_list = img_list['sharp'] #+ img_list['sharp']
        self.crop_size = crop_size
        self.path = path
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.4462 , 0.4375, 0.4244],  # mean
                                                                    std=[0.2317 , 0.2266, 0.2302])])
        self.init_len = len(img_list['blur'])


    def crop_resize_totensor(self,img, crop_location ):
        img256 = img.crop( crop_location )
        return self.to_tensor(img256)

    def __len__(self):
        return len(self.blur_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.blur_list[idx]
        clear_img_name = self.sharp_list[idx]

        blurry_img = Image.open(self.path + blurry_img_name[2:])
        clear_img = Image.open(self.path + clear_img_name[2:])
        assert blurry_img.size == clear_img.size


        i, j, h, w = transforms.RandomCrop.get_params(
            blurry_img, output_size=(256, 256))
        blurry_img = F.crop(blurry_img, i, j, h, w)
        clear_img = F.crop(clear_img, i, j, h, w)

        if random.random() < 0.5:
            blurry_img = F.hflip(blurry_img)
            clear_img = F.hflip(clear_img)

        crop_left = int(np.floor(np.random.uniform(0, blurry_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_img.size[1] - self.crop_size[1] + 1)))
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        img256 = self.crop_resize_totensor(blurry_img, crop_location)
        label256 = self.crop_resize_totensor(clear_img, crop_location)

        class_label = 0.0 if idx<self.init_len else 1.0
        batch = {'img256': img256, 'label256': label256, 'class_label': class_label}
        return batch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, path='./'):
        super(type(self), self).__init__()
        self.img_list = img_list
        self.path = path
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.4462, 0.4375, 0.4244],  # mean
                                                                  std=[0.2317, 0.2266, 0.2302])])

    def __len__(self):
        return len(self.img_list['blur'])

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.img_list['blur'][idx]
        clear_img_name = self.img_list['sharp'][idx]

        blurry_img = Image.open(self.path + blurry_img_name[2:])
        clear_img = Image.open(self.path + clear_img_name[2:])
        assert blurry_img.size == clear_img.size

        img256 = self.to_tensor(blurry_img)
        label256 = self.to_tensor(clear_img)
        batch = {'img256': img256, 'label256': label256, 'dir':blurry_img_name}

        return batch

