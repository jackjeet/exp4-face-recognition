# ==============================================
# 4_loss_optimizer_config.py
# 功能：完整配置损失函数（交叉熵）和优化器（Adam）
# 依赖：需确保data文件夹下有train/test数据集
# 适配低版本PyTorch，移除不支持的verbose参数
# ==============================================

# 步骤1：导入所有依赖库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob
import os

# 步骤2：复用第二步的数据集类（无需修改）
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

# 步骤3：复用第三步的PyTorch原生风格CNN模型（无需修改）
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

# 步骤4：加载数据集（复用第二步的预处理和划分逻辑）
# 数据预处理
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

# 数据集路径（请确认路径与你的实际路径一致！）
# 若你的train/test直接在项目根目录，修改为：train_data_dir = "./train"，test_data_dir = "./test"
train_data_dir = "./data/train"
test_data_dir = "./data/test"

# 加载原始数据集
raw_train_dataset = EmoreFaceDataset(img_dir=train_data_dir, transform=train_transform)
test_dataset = EmoreFaceDataset(img_dir=test_data_dir, transform=test_transform)

# 划分训练集和验证集（8:2）
train_size = int(0.8 * len(raw_train_dataset))
val_size = len(raw_train_dataset) - train_size
train_dataset, val_dataset = random_split(raw_train_dataset, [train_size, val_size])

# 构建数据加载器
batch_size = 32
num_workers = 0  # Windows系统固定设为0，避免报错
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 步骤5：配置运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 步骤6：初始化模型并迁移到设备
model = PyTorch_Native_CNN(num_classes=2)
model.to(device)

# 步骤7：配置交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 步骤8：配置Adam优化器
optimizer = optim.Adam(
    params=model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

# 步骤9：配置学习率调度器（移除verbose参数，适配低版本PyTorch）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

# 步骤10：打印配置结果（验证是否成功）
print("="*60)
print("第四步：损失函数和优化器配置完成！")
print("="*60)
print(f"运行设备：{device}")
print(f"损失函数：{criterion.__class__.__name__}")
print(f"优化器：{optimizer.__class__.__name__}")
print(f"初始学习率：{optimizer.param_groups[0]['lr']}")
print(f"L2正则化系数：{optimizer.param_groups[0]['weight_decay']}")
print(f"学习率调度器：{scheduler.__class__.__name__}（已启用）")
print("-"*60)
print(f"训练集样本数：{len(train_dataset)}")
print(f"验证集样本数：{len(val_dataset)}")
print(f"测试集样本数：{len(test_dataset)}")
print(f"训练集批次数量：{len(train_loader)}")
print(f"验证集批次数量：{len(val_loader)}")
print(f"测试集批次数量：{len(test_loader)}")
print("="*60)