# ==============================================
# 步骤1：导入 PyTorch 依赖库（无需 TensorFlow/Keras）
# ==============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  # 查看模型结构（可选）


# ==============================================
# 模型1：PyTorch 原生风格 CNN（继承 nn.Module）
# 满足：卷积层+全连接层+激活函数+前向传播
# ==============================================
class PyTorch_Native_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PyTorch_Native_CNN, self).__init__()

        # 卷积层组（3层卷积+池化+Dropout）
        self.conv_layers = nn.Sequential(
            # 卷积层1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
            nn.Dropout(0.25),  # 防止过拟合

            # 卷积层2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 卷积层3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # 全连接层组（2层全连接+Dropout）
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),  # 112/2^3=14，展平后维度
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)  # 输出层
        )

    # 前向传播方法（核心：定义数据流动路径）
    def forward(self, x):
        x = self.conv_layers(x)  # 卷积特征提取
        x = x.view(-1, 128 * 14 * 14)  # 展平
        x = self.fc_layers(x)  # 全连接层分类
        return x


# ==============================================
# 模型2：PyTorch 简洁风格 CNN（模拟 Keras Sequential）
# 满足：卷积层+全连接层+激活函数+前向传播（自动顺序执行）
# ==============================================
def PyTorch_Simple_CNN(num_classes=2):
    # 完全模拟 Keras Sequential 顺序结构，代码简洁
    model = nn.Sequential(
        # 卷积层1
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.25),

        # 卷积层2
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.25),

        # 卷积层3
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.25),

        # 展平层（连接卷积和全连接）
        nn.Flatten(),  # PyTorch 1.7+ 支持，无需手动计算维度

        # 全连接层1
        nn.Linear(128 * 14 * 14, 512),
        nn.ReLU(),
        nn.Dropout(0.5),

        # 输出层
        nn.Linear(512, num_classes)
    )
    # PyTorch Sequential 自动实现前向传播（按层顺序执行）
    return model


# ==============================================
# 步骤2：验证两个模型（确保满足需求）
# ==============================================
if __name__ == "__main__":
    # 检测运行设备（CPU/GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")
    print("=" * 60)

    # ---------------------- 验证模型1：PyTorch 原生风格 ----------------------
    print("模型1：PyTorch 原生风格 CNN")
    model1 = PyTorch_Native_CNN(num_classes=2).to(device)
    # 查看模型结构
    try:
        summary(model1, input_size=(3, 112, 112))
    except:
        print("跳过结构打印（需安装 torchsummary：pip install torchsummary）")
    # 测试前向传播
    test_input1 = torch.randn(32, 3, 112, 112).to(device)
    test_output1 = model1(test_input1)
    print(f"前向传播输出形状：{test_output1.shape}")  # 预期 (32, 2)
    print("模型1 验证成功！")
    print("=" * 60)

    # ---------------------- 验证模型2：PyTorch 简洁风格 ----------------------
    print("\n模型2：PyTorch 简洁风格 CNN（模拟 Keras）")
    model2 = PyTorch_Simple_CNN(num_classes=2).to(device)
    # 查看模型结构
    try:
        summary(model2, input_size=(3, 112, 112))
    except:
        print("跳过结构打印（需安装 torchsummary：pip install torchsummary）")
    # 测试前向传播
    test_input2 = torch.randn(32, 3, 112, 112).to(device)
    test_output2 = model2(test_input2)
    print(f"前向传播输出形状：{test_output2.shape}")  # 预期 (32, 2)
    print("模型2 验证成功！")
    print("=" * 60)