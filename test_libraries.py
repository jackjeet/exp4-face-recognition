# 测试所需库是否安装
try:
    import torch
    print("✅ torch 已安装，版本：", torch.__version__)
except ImportError:
    print("❌ torch 未安装")

try:
    import torchvision
    print("✅ torchvision 已安装，版本：", torchvision.__version__)
except ImportError:
    print("❌ torchvision 未安装")

try:
    import numpy
    print("✅ numpy 已安装，版本：", numpy.__version__)
except ImportError:
    print("❌ numpy 未安装")

try:
    import matplotlib
    print("✅ matplotlib 已安装，版本：", matplotlib.__version__)
except ImportError:
    print("❌ matplotlib 未安装")

try:
    import cv2
    print("✅ opencv-python 已安装，版本：", cv2.__version__)
except ImportError:
    print("❌ opencv-python 未安装")