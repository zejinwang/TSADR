import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize


from data import util
import random


def is_image_file(filename):
		return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def transform(crop):
    if crop is not None:
        return transforms.Compose([
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])
    else:
        return transforms.Compose([
                transforms.ToTensor()
            ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDBreader(Dataset):
    """
    DBreader reads all triplet set of frames in a directory.
    Each triplet set contains frame 0, 1, 2.
    Each image is named frame0.png, frame1.png, frame2.png.
    Frame 0, 2 are the input and frame 1 is the output.
    """

    def __init__(self, db_dir, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        #self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])
        triplet_list = []
        with open(db_dir) as f:
            for line in f.readlines():
                line = line.strip()  #去除每行头尾空格和换行符
                if not len(line) or line.startswith('#'):
                    continue
                triplet_list.append(line)
        self.triplet_list = triplet_list
        self.file_len = len(self.triplet_list)
        self.dir = db_dir.split('_')[0] + '_triplet/sequences/'
    def __getitem__(self, index):
        frame0 = self.transform(Image.open(self.dir + self.triplet_list[index] + "/im1.png"))
        frame1 = self.transform(Image.open(self.dir + self.triplet_list[index] + "/im2.png"))
        frame2 = self.transform(Image.open(self.dir + self.triplet_list[index] + "/im3.png"))

        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len


class ValDBreader(Dataset):
    """
    DBreader reads all triplet set of frames in a directory.
    Each triplet set contains frame 0, 1, 2.
    Each image is named frame0.png, frame1.png, frame2.png.
    Frame 0, 2 are the input and frame 1 is the output.
    """

    def __init__(self, db_dir, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        #self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])
        triplet_list = []
        with open(db_dir) as f:
            for line in f.readlines():
                line = line.strip()  #去除每行头尾空格和换行符
                if not len(line) or line.startswith('#'):
                    continue
                triplet_list.append(line)
        self.triplet_list = triplet_list
        #with open(db_dir) as f:
        #    lines = f.readlines()
        #    for index in range(len(lines)):
        #        lines[index] = lines[index].replace('\n','')
        #self.triplet_list = lines
        self.file_len = len(self.triplet_list)
        self.dir = db_dir.split('_')[0] + '_triplet/sequences/'
    def __getitem__(self, index):
        frame0 = self.transform(Image.open(self.dir + self.triplet_list[index] + "/im1.png"))
        frame1 = self.transform(Image.open(self.dir + self.triplet_list[index] + "/im2.png"))
        frame2 = self.transform(Image.open(self.dir + self.triplet_list[index] + "/im3.png"))

        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, RCrop=(256,256)):
        super(TrainDatasetFromFolder, self).__init__()  # 执行父类的__init__方法
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        #self.image_filenames.sort(key=lambda x: int(x.split('_')[3]) * 10 + int(x.split('_')[4].split('.')[0]))
        self.image_filenames = sorted(self.image_filenames, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[-4:] )

        self.crop_h, self.crop_w = RCrop 
    def __getitem__(self, index):

        img_0 = util.read_img(None, self.image_filenames[index*3])
        img_1 = util.read_img(None, self.image_filenames[index*3+1])
        img_2 = util.read_img(None, self.image_filenames[index*3+2])
        
        H, W, _ = img_0.shape
        # randomly crop
        rnd_h = random.randint(0, max(0, H - self.crop_h))
        rnd_w = random.randint(0, max(0, W - self.crop_w))
        img_0 = img_0[rnd_h : rnd_h + self.crop_h, rnd_w : rnd_w + self.crop_w, :]
        img_1 = img_1[rnd_h : rnd_h + self.crop_h, rnd_w : rnd_w + self.crop_w, :]
        img_2 = img_2[rnd_h : rnd_h + self.crop_h, rnd_w : rnd_w + self.crop_w, :]

        # augmentation - flip, rotate
        img_0, img_1, img_2 = util.augment([img_0, img_1, img_2],True,True)
         
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_0.shape[2] == 3:
            img_0 = img_0[:, :, [2, 1, 0]]
            img_1 = img_1[:, :, [2, 1, 0]]
            img_2 = img_2[:, :, [2, 1, 0]]
        img_0 = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_0, (2, 0, 1)))
        ).float()
        img_1 = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_1, (2, 0, 1)))
        ).float()
        img_2 = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_2, (2, 0, 1)))
        ).float()

        return img_0, img_1, img_2

    def __len__(self):
        # print(len(self.image_filenames)//3)
        return len(self.image_filenames) // 3


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        #self.image_filenames.sort(key=lambda x: int(x.split('_')[3]) * 10 + int(x.split('_')[4].split('.')[0]))
        #self.image_filenames = sorted(self.image_filenames, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[-4:] )
        self.image_filenames = sorted(self.image_filenames, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[-4:] )

    def __getitem__(self, index):
        frame0 = ToTensor()(Image.open(self.image_filenames[index * 3]))
        frame1 = ToTensor()(Image.open(self.image_filenames[index * 3 + 1]))
        frame2 = ToTensor()(Image.open(self.image_filenames[index * 3 + 2]))

        frame0 = torch.cat([frame0,frame0,frame0],0)
        frame1 = torch.cat([frame1,frame1,frame1],0)
        frame2 = torch.cat([frame2,frame2,frame2],0)

        return frame0, frame1, frame2

    def __len__(self):
        return len(self.image_filenames) // 3


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        #self.image_filenames.sort(key=lambda x: int(x.split('_')[3]) * 10 + int(x.split('_')[4].split('.')[0]))
        self.image_filenames = sorted(self.image_filenames, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[-4:])

    def __getitem__(self, index):
        img_name = self.image_filenames[index * 3].split('/')[-1]
        frame0 = ToTensor()(Image.open(self.image_filenames[index * 3]))
        frame1 = ToTensor()(Image.open(self.image_filenames[index * 3 + 1]))
        frame2 = ToTensor()(Image.open(self.image_filenames[index * 3 + 2]))

        frame0 = torch.cat([frame0,frame0,frame0],0)
        frame1 = torch.cat([frame1,frame1,frame1],0)
        frame2 = torch.cat([frame2,frame2,frame2],0)

        return img_name, frame0, frame1, frame2

    def __len__(self):
        return len(self.image_filenames) // 3
