

# 步骤1：导入依赖库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob
import os
import time
import numpy as np

# 步骤2：复用数据集类和模型类
class EmoreFaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(img_dir, "*/*.jpg")) + glob.glob(os.path.join(img_dir, "*/*.png"))
        self.class_names = sorted(os.listdir(img_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        self.labels = [self.class_to_idx[os.path.basename(os.path.dirname(path))] for path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class PyTorch_Native_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PyTorch_Native_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.fc_layers(x)
        return x

# 步骤3：超参数配置
EPOCHS = 1  # 1轮快速训练
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# 固定配置（重点：模型保存路径）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DATA_DIR = "./data/train"  # 确保路径正确
TEST_DATA_DIR = "./data/test"
CHECKPOINT_DIR = "./checkpoints"  # 模型保存文件夹
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "final_model_1epoch.pth")  # 最终模型路径
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # 自动创建文件夹（不存在则创建）

# 步骤4：加载数据集
train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

raw_train_dataset = EmoreFaceDataset(img_dir=TRAIN_DATA_DIR, transform=train_transform)
test_dataset = EmoreFaceDataset(img_dir=TEST_DATA_DIR, transform=test_transform)
train_size = int(0.8 * len(raw_train_dataset))
val_size = len(raw_train_dataset) - train_size
train_dataset, val_dataset = random_split(raw_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 步骤5：初始化模型、损失函数、优化器
model = PyTorch_Native_CNN(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

# 步骤6：训练/验证函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播+参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

        # 每50批打印进度
        if (batch_idx + 1) % 50 == 0:
            batch_loss = loss.item()
            batch_acc = (predicted == labels).sum().item() / images.size(0)
            print(f"  批次[{batch_idx+1}/{len(loader)}] - 损失: {batch_loss:.4f} - 准确率: {batch_acc:.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计指标
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# 步骤7：主训练循环（1轮）+ 模型保存
print("="*60)
print(f"开始1轮快速训练 + .pth模型保存 - 设备: {DEVICE}")
print(f"模型最终保存路径: {MODEL_SAVE_PATH}")
print("="*60)

train_log = {"loss": [], "acc": []}
val_log = {"loss": [], "acc": []}
start_time = time.time()

# 1轮训练
for epoch in range(EPOCHS):
    print(f"\n【Epoch {epoch+1}/{EPOCHS}】")
    print("-"*40)

    # 训练+验证
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)

    # 调整学习率
    scheduler.step(val_loss)

    # 记录日志
    train_log["loss"].append(train_loss)
    train_log["acc"].append(train_acc)
    val_log["loss"].append(val_loss)
    val_log["acc"].append(val_acc)

    # 打印本轮结果
    print(f"\n训练集 - 平均损失: {train_loss:.4f} | 准确率: {train_acc:.4f}")
    print(f"验证集 - 平均损失: {val_loss:.4f} | 准确率: {val_acc:.4f}")
    print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # ---------------------- 核心：保存.pth模型文件 ----------------------
    # 保存内容：模型权重 + 优化器状态 + 训练日志 + 关键指标（便于后续复现）
    torch.save({
        "epoch": epoch + 1,  # 训练轮数
        "model_state_dict": model.state_dict(),  # 模型权重（核心）
        "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态（便于继续训练）
        "train_loss": train_loss,  # 训练集损失
        "train_acc": train_acc,  # 训练集准确率
        "val_loss": val_loss,  # 验证集损失
        "val_acc": val_acc,  # 验证集准确率
        "train_log": train_log,  # 完整训练日志
        "val_log": val_log,  # 完整验证日志
        "hyper_params": {  # 超参数（便于复现）
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY
        }
    }, MODEL_SAVE_PATH)

    print(f"\n✅ 模型已保存为.pth文件！路径：{MODEL_SAVE_PATH}")
    print(f"保存内容包含：模型权重、优化器状态、训练日志、超参数")

# 训练结束统计
end_time = time.time()
total_time = (end_time - start_time) / 60
print("\n" + "="*60)
print("1轮训练+模型保存完成！")
print(f"总训练时间: {total_time:.2f} 分钟")
print(f"验证集最终准确率: {val_acc:.4f}")
print(f"模型文件: {MODEL_SAVE_PATH}")
print("="*60)

# 步骤8：如何加载.pth模型（后续使用指南）
print("\n【加载.pth模型的方法】")
print("后续若需使用保存的模型，可运行以下代码：")
print("="*60)
print('import torch')
print('from 5_model_training import PyTorch_Native_CNN  # 需导入模型类')
print('')
print('# 初始化模型')
print('model = PyTorch_Native_CNN(num_classes=2)')
print(f'checkpoint = torch.load("{MODEL_SAVE_PATH}")  # 加载.pth文件')
print('model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型权重')
print('model.eval()  # 设为评估模式（禁用Dropout）')
print('')
print('# 可选：加载优化器状态（继续训练）')
print('# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])')
print('# 可选：查看保存的指标和日志')
print('# print("验证集准确率:", checkpoint["val_acc"])')
print('# print("训练日志:", checkpoint["train_log"])')
print("="*60)