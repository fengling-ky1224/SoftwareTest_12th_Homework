# predict.py
import os
import numpy as np
import skimage.io as io
import torch

import Attention_Unet_Pytorch.models.unet as unet
import Attention_Unet_Pytorch.utils.utils as utils

# 使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预测图像处理函数
def preprocess_image(file_path, size=512):
    img = np.array(io.imread(file_path)).astype(np.float32) / 255.0
    img = utils.contrast_and_reshape(img)  # 增强对比度
    img = img.reshape(1, 1, size, size)
    return torch.tensor(img, dtype=torch.float32)

# 加载模型
def load_model(path, filters=8, attention=True):
    model = unet.Unet(filters=filters, attention=attention).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# 获取文件路径列表
def get_image_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))]

# 主预测逻辑
def predict_folder(model_path, image_folder, output_folder, size=512):
    os.makedirs(output_folder, exist_ok=True)
    model = load_model(model_path)

    image_files = get_image_files(image_folder)
    for img_path in image_files:
        img_tensor = preprocess_image(img_path, size=size).to(device)
        with torch.no_grad():
            output, _ = model(img_tensor)
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        io.imsave(save_path, mask)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # 配置你的路径
    model_path = "saved_models/net.pth"
    input_folder = r"D:\Users\86188\Desktop\BUSI-256\images"
    output_folder = r"D:\Users\86188\Desktop\BUSI-256\predictions"

    predict_folder(model_path, input_folder, output_folder, size=512)