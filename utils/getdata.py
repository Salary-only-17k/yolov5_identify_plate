import os
import pathlib
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader as DataLoader

"""https://github.com/we0091234/yolov7_plate
"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",  "2",
  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",  "H",  "J",  "K",
  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",  "V",  "W",  "X",  "Y",  "Z",  "港", "学", "使",
  "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航", "空"

"""

class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def transfroms(input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.CenterCrop((input_size, input_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


class Dataset(data.Dataset): 
    def __init__(self, mode, dir:str,indx:int=2):  
        self.mode = mode
        self.format =format
        self.list_img = []  
        self.list_label= [] 
        self.transform = transfroms()  
        assert self.mode in ['train', 'val', 'test'],ValueError
        dir = dir + f'/{self.mode}/'  
        for filepth in list(pathlib.Path(dir).glob("**/*.png"))+list(pathlib.Path(dir).glob("**/*.jpg")):  
            filepth = str(filepth)
            self.list_img.append(filepth)  
            filename = os.path.basename(filepth)
            label_pool = filename.split('.')
            label = [int(i) for i in label_pool.split('_')]
            self.list_label+=float(label)
        self.data_size = len(self.list_label)           
       
    def __doc__(self):
        print("0123456.jpg")

    def __getitem__(self, item):  
        if self.mode == 'train':  
            img = Image.open(self.list_img[item]) 
            label = self.list_label[item]  
            return self.transform['train'](img),torch.LongTensor([label])  
        elif self.mode in ['val', 'test']:  
            img = Image.open(self.list_img[item])  
            label = self.list_label[item]  
            return self.transform['test'](img), torch.LongTensor([label]) 
        else:
            print('None')

    def __len__(self):
        return self.data_size  # 返回数据集大小


# if __name__ == '__main__':
#     print(transfroms()['train'])
#     dataset_dir = r'../data'  # 数据集路径
#     # a, [b, c] = next(iter(MulitDataset('train', dataset_dir)))
#     # print(a)
#     # print(b)
#     # print(c)
#     print('-' * 20)
#     from torch.utils.data import DataLoader as DataLoader

#     # test_data = MulitDataset('train', dataset_dir)
#     test_data = HoTDataset('train', dataset_dir)
#     dataset = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

#     tmp = []
#     for d, l in dataset:
#         print(l)
#         tmp.append(len(d))
#     print(sum(tmp), "  ", len(dataset))

#     # a, [b, c] = next(iter(Dataset('test', dataset_dir)))
#     # print(a, b, c)
#     # a,[b,c] =Dataset('train', dataset_dir)
#     # print(a,b,c)
