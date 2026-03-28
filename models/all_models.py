import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
#  ----------------------- CNN for rf and videos------------------
class ConvBNReLU3D(nn.Module):
    """3D convolution + BatchNorm + ReLU block"""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Modality3DCNN(nn.Module):
    """Generic 3D CNN for video or RF frames"""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # 3D CNN blocks
        self.conv1 = ConvBNReLU3D(in_channels, 32, stride=1)
        self.conv2 = ConvBNReLU3D(32, 64, stride=2)
        self.conv3 = ConvBNReLU3D(64, 128, stride=2)
        self.conv4 = ConvBNReLU3D(128, 256, stride=2)

        # Adaptive pooling → fixed-size feature vector
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes - 1),  # binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]
               - T = number of frames (video or RF segment)
               - C = channels (3 for RGB, 1 for grayscale)
        """
        if x.ndim != 5:
            raise ValueError("Input tensor must be 5D: [B, T, C, H, W]")

        # Convert to [B, C, T, H, W] for nn.Conv3d
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global average pooling
        x = self.avg_pool(x)  # [B, 256, 1, 1, 1]
        x = torch.flatten(x, 1)  # [B, 256]

        out = self.fc(x)
        return out.squeeze()






###  ----------------------------MobileNet for videos --------------------------------------

class ConvBNReLU3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Visual_MobileNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv1 = ConvBNReLU3D(3, 32, stride=2)
        self.conv_blocks = nn.Sequential(
            ConvBNReLU3D(32, 64, stride=2),
            ConvBNReLU3D(64, 128, stride=2),
            ConvBNReLU3D(128, 128),
            ConvBNReLU3D(128, 256, stride=2),
            ConvBNReLU3D(256, 256),
            ConvBNReLU3D(256, 512, stride=2),
            *[ConvBNReLU3D(512, 512) for _ in range(5)],
            ConvBNReLU3D(512, 1024, stride=2),
            ConvBNReLU3D(1024, 1024),
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # NOTE: This matches the feature map we expose as out_5 (256 channels in our staged outputs below)
        self.fc = nn.Sequential(
            nn.Linear(256, n_classes - 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv1(x)

        out_1 = self.conv_blocks[0](x)
        out_2 = self.conv_blocks[1](out_1)
        out_3 = self.conv_blocks[2](out_2)
        out_4 = self.conv_blocks[3](out_3)
        out_5 = self.conv_blocks[4](out_4)

        x = self.avg_pool(out_5)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return  out.squeeze()

  
  #  -------------------------------------------------  CNN for audio ------------------------------------------------
class LogMelCNN_64(nn.Module):
    def __init__(self):
        super(LogMelCNN_64, self).__init__()

        # Conv Block 1: 64x40 -> 32x20 (after pool)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Block 2: 32x20 -> 16x10 (after pool)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Block 3: 16x10 -> 8x5 (after pool)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)

        # Flattened size: 128 channels * 8 height * 5 width = 5120
        self.fc1 = nn.Linear(128 * 8 * 5, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, 1, 64, 40]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.classifier(x)
        return out.squeeze()


#  ----------- AudioVGG19 audio------------------------------

class GlobalPooling2D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), x.size(1), -1)
        return torch.mean(x, dim=2)


class AudioVGG19(nn.Module):
    """
    VGG19 features + global average pooling + batch norm + linear head.
    Returns logits for binary classification.
    """

    def __init__(self):
        super().__init__()
        vgg_features = list(tv_models.vgg19(weights=None).features)
        vgg_features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.features = nn.Sequential(*vgg_features)
        self.global_pool = GlobalPooling2D()
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.bn(x)
        x = self.fc(x)
        return x.squeeze(1)


# ---------------------------- ResNet18 for RF -------------------------

class RF_ResNet18(nn.Module):
    def __init__(self, n_classes=2, pretrained=True, freeze_backbone=True):
        super(RF_ResNet18, self).__init__()
        
        # 1. تحميل الموديل بنفس الطريقة (Weights Default)
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = tv_models.resnet18(weights=weights)
        
        # 2. تجميد الأوزان (اختياري كما في الكود الخاص بك) لعمل Transfer Learning
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 3. تغيير الطبقة الأخيرة لتناسب مشروع SAQR (Drone vs No Drone)
        # نستخدم n_classes - 1 (أي 1) مع Sigmoid للـ Binary Classification
        # أو n_classes (أي 2) إذا كنت ستستخدم CrossEntropyLoss
        # لكي يتوافق مع كود الـ Fusion السابق، سنستخدم مخرجاً واحداً مع Sigmoid
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, n_classes - 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Input x shape: (Batch, 3, H, W)
        """
        # ResNet18 تتوقع مدخل 4D (صورة أو Spectrogram)
        out = self.model(x)
        return out.squeeze() # لإرجاع احتمالية واحدة لكل Batch






     
     
