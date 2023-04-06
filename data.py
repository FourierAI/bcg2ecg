import os

import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, str_):
        self.bcgs = self.__loaddata(str_)

    def __loaddata(self, str_):
        bcgs = []
        for line in str_.split('\n'):
            if line:
                bcg = [float(i) for i in line.split(',')]
                bcgs.append(bcg)
        return bcgs

    # 返回数据集大小
    def __len__(self):
        return len(self.bcgs)

    # 得到数据内容和标签
    def __getitem__(self, index):
        bcgs = torch.Tensor(self.bcgs[index])
        return bcgs

if __name__ == '__main__':
    content = open('slp01a.csv').read()
    data = Dataset(content)