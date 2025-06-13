import os
import numpy as np
import progressbar
import skimage.io as io
import sklearn.model_selection as sk
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import Attention_Unet_Pytorch.models.unet as unet
import Attention_Unet_Pytorch.utils.data as data
import Attention_Unet_Pytorch.utils.utils as utils

# CPU 模式（禁用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保保存目录存在
# SAVE_PATH = "saved_models/net.pth"
SAVE_PATH = "saved_models/net2.pth"
LOSS = np.inf

def save_model(net, loss):
    global LOSS
    if loss < LOSS:
        LOSS = loss
        # 确保保存目录存在
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(net.state_dict(), SAVE_PATH)

# 进度条
from tqdm import tqdm  # 导入 tqdm

widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]

# BASE_PATH = r"D:\Users\86188\Desktop\BUSI-256\\"
BASE_PATH = r"D:\Users\86188\Desktop\isic2018\train\\"

def get_datasets(path_img, path_label, config):
    img_path_list = utils.list_files_path(path_img)
    label_path_list = utils.list_files_path(path_label)
    img_path_list, label_path_list = utils.shuffle_lists(img_path_list, label_path_list)
    img_train, img_val, label_train, label_val = sk.train_test_split(
        img_path_list, label_path_list, test_size=0.2, random_state=42
    )
    dataset_train = data.JB_Dataset(
        config.batch_size, config.size, img_train, label_train
    )
    dataset_val = data.JB_Dataset(config.batch_size, config.size, img_val, label_val)
    return dataset_train, dataset_val

def train(path_imgs, path_labels, config, epochs=5):
    net = unet.Unet(config.filters).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    criterion = nn.BCELoss()

    dataset_train, dataset_val = get_datasets(path_imgs, path_labels, config)
    train_loss = []
    val_loss = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_val_loss = []
        utils.print_gre(f"Epoch {epoch + 1}/{epochs}")

        with tqdm(total=len(dataset_train), desc="Training", ncols=100) as bar:
            net.train()
            for i in range(len(dataset_train)):
                bar.update(1)
                imgs, labels = dataset_train[i]
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                output, _ = net(imgs)
                loss_train = criterion(output, labels)
                loss_train.backward()
                optimizer.step()
                epoch_train_loss.append(loss_train.item())

        train_loss.append(np.array(epoch_train_loss).mean())

        with tqdm(total=len(dataset_val), desc="Validation", ncols=100) as bar2:
            net.eval()
            for j in range(len(dataset_val)):
                bar2.update(1)
                imgs, labels = dataset_val[j]
                imgs, labels = imgs.to(device), labels.to(device)
                output, _ = net(imgs)
                loss_val = criterion(output, labels)
                epoch_val_loss.append(loss_val.item())

        val_loss_epoch = np.array(epoch_val_loss).mean()
        val_loss.append(val_loss_epoch)

        utils.print_gre(f"Loss train {np.array(epoch_train_loss).mean()}\nLoss val {val_loss_epoch}")
        scheduler.step()
        save_model(net, val_loss_epoch)

    utils.learning_curves(train_loss, val_loss)

def learning_curves(train, val):
    # 创建文件夹
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.suptitle("Training Curves")
    ax.plot(train, label="Train Loss")
    ax.plot(val, label="Validation Loss")
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.legend()

    save_path = os.path.join(save_dir, "isic_loss_curve.png")  # 设置文件名
    fig.savefig(save_path)  # 保存图像
    plt.close(fig)

if __name__ == "__main__":
    args = utils.get_args()
    train(
        BASE_PATH + "images\\",
        BASE_PATH + "masks\\",
        config=args,
        epochs=args.epochs,
    )