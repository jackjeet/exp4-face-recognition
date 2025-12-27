import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob


# 1. 定义数据集类：读取图像和对应标签
class EmoreFaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 遍历所有.jpg/.png格式图像（支持常见图片格式）
        self.img_paths = glob.glob(os.path.join(img_dir, "*/*.jpg")) + glob.glob(os.path.join(img_dir, "*/*.png"))
        # 提取类别（子文件夹名作为类别名，映射为数字标签）
        self.class_names = sorted(os.listdir(img_dir))  # 排序保证类别顺序固定
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        self.labels = [self.class_to_idx[os.path.basename(os.path.dirname(path))] for path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)  # 返回数据集总样本数

    def __getitem__(self, idx):
        # 读取单张图像和标签
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # 统一转为RGB（避免灰度图报错）

        if self.transform:
            image = self.transform(image)  # 应用预处理/增强
        return image, label  # 返回（图像张量，标签）


# 2. 定义预处理与数据增强规则
# 训练集：含增强（提升模型泛化能力）
train_transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 人脸任务常用尺寸，统一调整
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色扰动
    transforms.ToTensor(),  # 转为PyTorch张量（像素值0-1）
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到[-1,1]
])

# 测试集：无增强（仅统一格式，保证评估准确）
test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. 加载数据集并划分（训练集→训练集+验证集）
# 数据集路径（相对项目根目录，无需改绝对路径）
train_data_dir = "./data/train"
test_data_dir = "./data/test"

# 加载原始训练集和测试集
raw_train_dataset = EmoreFaceDataset(img_dir=train_data_dir, transform=train_transform)
test_dataset = EmoreFaceDataset(img_dir=test_data_dir, transform=test_transform)

# 划分训练集为「训练集（80%）+ 验证集（20%）」（验证集用于监测训练效果）
train_size = int(0.8 * len(raw_train_dataset))
val_size = len(raw_train_dataset) - train_size
train_dataset, val_dataset = random_split(raw_train_dataset, [train_size, val_size])

# 4. 构建数据加载器（批量读取，用于模型训练）
batch_size = 32  # 批大小：性能好改64，性能差改16
num_workers = 0  # 多线程：Windows报错则设为0，Mac/Linux设2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 训练集打乱，提升训练效果
    num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # 验证集不打乱
    num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # 测试集不打乱
    num_workers=num_workers
)

# 5. 验证结果（打印数据集信息，确认加载成功）
print("=" * 50)
print("数据集处理完成！")
print(f"训练集样本数：{len(train_dataset)}")
print(f"验证集样本数：{len(val_dataset)}")
print(f"测试集样本数：{len(test_dataset)}")
print(f"人脸类别数：{len(raw_train_dataset.class_names)}")
print(f"类别列表：{raw_train_dataset.class_names[:10]}...")  # 打印前10个类别示例
print("=" * 50)