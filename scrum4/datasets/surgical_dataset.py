import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
# from utils import get_transform


PATH = 'C:\\Users\\gmita\\Desktop\\work_3_2_2565\\onboard\\WS-DAN.PyTorch-master\\datasets\\Surgical_tools'

import torchvision.transforms as transforms
def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class Surgical():
    def __init__(self, phase='train', resize=(448, 448)):
        self.num_classes = 4
        self.input_size = resize
        self.is_train = phase
        self.classes = ["Curved Mayo Scissor","Scalpel","Straight Dissection Clamp","Straight Mayo Scissor"]
        img_path = os.path.join(PATH, 'images')
        train_label_file = open(os.path.join(PATH, 'train.txt'))
        val_label_file = open(os.path.join(PATH, 'val.txt'))
        test_label_file = open(os.path.join(PATH, 'test.txt'))
        Image = []

        if self.is_train == 'train':
            for line in train_label_file:
                Image.append([os.path.join(img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])])

        elif self.is_train == 'val':  
            for line in val_label_file:
                Image.append([os.path.join(img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])])

        else :
            for line in test_label_file:
                Image.append([os.path.join(img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])])

        self.image = Image

        self.transform = get_transform(self.input_size, self.is_train)

    def __getitem__(self, index):
        image, target = Image.open(os.path.join(PATH, 'images', self.image[index][0])).convert('RGB'), self.image[index][1]
        image = self.transform(image)
        # print("hi")


        return image, target

    def __len__(self):
        return len(self.image)
        
if __name__ == '__main__':
    ds = Surgical(phase = 'train')
    ew = Surgical(phase = 'val')
    bb = Surgical(phase = 'test')
    print(len(ds))