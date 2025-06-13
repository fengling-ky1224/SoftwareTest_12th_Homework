import os
import numpy as np
import skimage.io as io

# 文件夹路径
image_folder = r"D:\Users\86188\Desktop\BUSI-256\images"
mask_folder = r"D:\Users\86188\Desktop\BUSI-256\masks"

# 判断图像是否为 RGB 图像或灰度图
def check_image_type(image_path):
    img = io.imread(image_path)

    if img.ndim == 3 and img.shape[2] == 3:
        return "RGB"
    elif img.ndim == 2:
        return "Grayscale"
    else:
        return "Unknown"

# 处理所有图像
def process_images_in_folder(image_folder):
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        if img_path.endswith(('.png', '.jpg')):  # 确保是图像文件
            image_type = check_image_type(img_path)  # 检查图像类型
            print(f"{img_name} is {image_type}")

# 调用函数
process_images_in_folder(image_folder)