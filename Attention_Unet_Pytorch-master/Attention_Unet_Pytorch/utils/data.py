import numpy as np
import skimage.io as io
import skimage.transform as transform  # 用于调整图像大小
import torch
from torch.utils.data import Dataset

# 图像对比度调整函数（可选）
def contrast_and_reshape(img):
    from skimage.exposure import equalize_adapthist
    img_adapteq = equalize_adapthist(img, clip_limit=0.03)
    return np.array(img_adapteq)

class JB_Dataset(Dataset):
    def __init__(
        self, batch_size, img_size, input_img_paths, label_img_paths, contrast=True
    ):
        self.batch_size = batch_size
        self.img_size = img_size  # 默认设置为 512
        self.input_img_paths = input_img_paths
        self.label_img_paths = label_img_paths
        self.contrast = contrast
        assert len(input_img_paths) == len(label_img_paths)
        print(f"Nb of images : {len(input_img_paths)}")

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_label_img_paths = self.label_img_paths[i : i + self.batch_size]
        x = np.zeros(
            (self.batch_size, 1, self.img_size, self.img_size), dtype="float32"
        )
        for j, path in enumerate(batch_input_img_paths):
            img = np.array(io.imread(path)) / 255  # 读取并归一化图像
            if img.ndim == 3:  # 如果是 RGB 图像，转换为灰度图
                img = np.dot(img[... , :3], [0.2989, 0.587, 0.114])  # 加权平均转换为灰度
            if self.contrast:
                img = contrast_and_reshape(img)
            img = transform.resize(img, (self.img_size, self.img_size), mode='reflect')  # 调整大小为 512x512
            x[j] = np.expand_dims(img, 0)  # 添加一个通道维度

        y = np.zeros(
            (self.batch_size, 1, self.img_size, self.img_size), dtype="float32"
        )
        for k, path_lab in enumerate(batch_label_img_paths):
            img = np.array(io.imread(path_lab)) / 255  # 读取并归一化掩码
            img = transform.resize(img, (self.img_size, self.img_size), mode='reflect')  # 调整大小为 512x512
            if img.ndim == 3:  # 如果掩码是 RGB，转换为单通道
                img = np.dot(img[... , :3], [0.2989, 0.587, 0.114])  # 加权平均转换为灰度
            y[k] = np.expand_dims(img, 0)  # 添加一个通道维度
        return torch.Tensor(x), torch.Tensor(y)