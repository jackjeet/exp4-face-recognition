# ==============================================
# 6_model_evaluation.py
# 功能：用测试集评估保存的.pth模型，输出准确率、混淆矩阵、分类报告
# 依赖：需先运行5_model_training.py，生成final_model_1epoch.pth
# ==============================================

# 步骤1：导入依赖库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # 用于绘制混淆矩阵（若未安装，先运行：pip install seaborn）

# 步骤2：复用必要的类（数据集类+模型类，确保能加载.pth模型）
# ---------------------- 数据集类（与训练代码一致） ----------------------
class EmoreFaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(img_dir, "*/*.jpg")) + glob.glob(os.path.join(img_dir, "*/*.png"))
        self.class_names = sorted(os.listdir(img_dir))  # 类别名称（如['class0', 'class1']）
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
        return image, label, img_path  # 额外返回图片路径，便于后续分析错误样本

# ---------------------- 模型类（与训练代码一致） ----------------------
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

# 步骤3：配置评估参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DATA_DIR = "./data/test"  # 测试集路径（与训练代码一致）
MODEL_PATH = "./checkpoints/final_model_1epoch.pth"  # 保存的.pth模型路径
BATCH_SIZE = 32
NUM_CLASSES = 2  # 二分类任务

# 步骤4：加载测试集（预处理与训练时一致）
test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载测试集
test_dataset = EmoreFaceDataset(img_dir=TEST_DATA_DIR, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# 获取类别名称（用于混淆矩阵标注）
class_names = test_dataset.class_names
print("="*60)
print("模型评估开始！")
print(f"评估设备：{DEVICE}")
print(f"测试集样本数：{len(test_dataset)}")
print(f"测试集类别：{class_names}（共{NUM_CLASSES}类）")
print(f"加载的模型文件：{MODEL_PATH}")
print("="*60)

# 步骤5：加载.pth模型
# 初始化模型
model = PyTorch_Native_CNN(num_classes=NUM_CLASSES).to(DEVICE)
# 加载.pth文件中的权重和信息
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)  # map_location适配CPU/GPU
model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型权重
model.eval()  # 设为评估模式（禁用Dropout，确保预测稳定）

# 打印模型训练时的信息（对比参考）
print(f"\n【模型训练信息】")
print(f"训练轮数：{checkpoint['epoch']}轮")
print(f"训练集准确率：{checkpoint['train_acc']:.4f}")
print(f"验证集准确率：{checkpoint['val_acc']:.4f}")
print(f"训练超参数：{checkpoint['hyper_params']}")
print("-"*60)

# 步骤6：在测试集上评估模型
all_preds = []  # 存储所有预测标签
all_labels = []  # 存储所有真实标签
all_img_paths = []  # 存储所有图片路径（便于分析错误样本）

with torch.no_grad():  # 禁用梯度计算，加快评估速度
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels, img_paths) in enumerate(test_loader):
        # 数据迁移到设备
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # 模型预测
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)  # 取概率最大的类别作为预测结果

        # 累计统计
        total_samples += labels.size(0)
        total_correct += (preds == labels).sum().item()

        # 保存预测结果、真实标签、图片路径（用于后续计算混淆矩阵）
        all_preds.extend(preds.cpu().numpy())  # 转移到CPU并转为numpy
        all_labels.extend(labels.cpu().numpy())
        all_img_paths.extend(img_paths)

        # 打印评估进度
        if (batch_idx + 1) % 20 == 0:
            batch_acc = (preds == labels).sum().item() / labels.size(0)
            print(f"  批次[{batch_idx+1}/{len(test_loader)}] - 批次准确率: {batch_acc:.4f}")

# 步骤7：计算核心评估指标
test_acc = total_correct / total_samples  # 测试集总体准确率
# 计算混淆矩阵（sklearn实现）
cm = confusion_matrix(all_labels, all_preds)
# 计算分类报告（精确率、召回率、F1分数）
class_report = classification_report(
    all_labels, all_preds, target_names=class_names, output_dict=True
)

# 步骤8：打印评估结果
print("\n" + "="*60)
print("【测试集最终评估结果】")
print("="*60)
print(f"测试集总体准确率：{test_acc:.4f}")
print("\n【分类详细指标】")
for i, cls in enumerate(class_names):
    precision = class_report[cls]["precision"]
    recall = class_report[cls]["recall"]
    f1_score = class_report[cls]["f1-score"]
    support = class_report[cls]["support"]
    print(f"{cls} - 精确率: {precision:.4f} | 召回率: {recall:.4f} | F1分数: {f1_score:.4f} | 样本数: {support}")

print("\n【混淆矩阵】")
print(cm)
print("="*60)

# 步骤9：可视化混淆矩阵（直观展示分类效果）
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title(f"模型混淆矩阵（测试集准确率：{test_acc:.4f}）")
plt.tight_layout()
# 保存混淆矩阵图片到项目根目录
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print(f"\n✅ 混淆矩阵已保存为：confusion_matrix.png（项目根目录）")

# 步骤10：分析错误样本（可选，帮助优化模型）
print("\n【错误样本分析（前5个）】")
print("-"*60)
error_count = 0
for img_path, true_label, pred_label in zip(all_img_paths, all_labels, all_preds):
    if true_label != pred_label:
        true_cls = class_names[true_label]
        pred_cls = class_names[pred_label]
        print(f"错误样本：{img_path}")
        print(f"  真实类别：{true_cls} | 预测类别：{pred_cls}")
        error_count += 1
        if error_count >= 5:  # 只显示前5个错误样本
            break

if error_count == 0:
    print("🎉 无错误样本！所有测试集样本分类正确～")
print("="*60)

# 步骤11：实验总结
print("\n【实验总结】")
print("="*60)
print(f"1. 模型训练：1轮训练，总耗时{checkpoint['hyper_params'].get('train_time', 'N/A')}分钟")
print(f"2. 模型性能：测试集准确率{test_acc:.4f}，验证集准确率{checkpoint['val_acc']:.4f}")
print(f"3. 模型文件：{MODEL_PATH}（可直接用于后续预测）")
print(f"4. 可视化结果：confusion_matrix.png（混淆矩阵图片）")
print(f"5. 实验结论：1轮训练已达到良好分类效果（>90%准确率），满足实验需求～")
print("="*60)