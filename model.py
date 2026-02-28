import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s


class Model02(nn.Module):
    """Pure EfficientNetV2-S adapter for ECG classification."""

    def __init__(self, num_classes):
        super().__init__()
        
        # EfficientNetV2-S backbone

        # EfficientNetV2-S backbone
        backbone = efficientnet_v2_s(weights=None)
        # Adapt first conv: 1 input channel (ECG grayscale 'image') instead of 3
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        # 2. Classifier (Pure EfficientNet)
        # We remove the metadata concatenation to match "Original" simple architecture.
        self.ecg_features = backbone.features
        self.ecg_pool = backbone.avgpool
        ecg_out_dim = backbone.classifier[-1].in_features  # 1280 for V2-S

        self.classifier = nn.Sequential(
            nn.Linear(ecg_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            # nn.Sigmoid(),  <-- Removed for Single-Label (CrossEntropyLoss expects logits)
        )

    def forward(self, x_meta, y_ecg):
        # x_meta is kept in signature for compatibility with training loop unpacking, but ignored.
        
        # y_ecg: (B, 12, 1000, 1) or similar.
        # We want to pad the '12' dimension to 32.
        # F.pad expects inputs (B, C, H, W) or similar if using 4D.
        
        # Move to (B, 1, 12, 1000) for Conv2d input (C_in=1, H=12, W=1000)
        # Permute: (B, 12, 1000, 1) -> (B, 1, 12, 1000)
        y = y_ecg.permute(0, 3, 1, 2) 
        
        # Pad H (12) to 32. 
        # F.pad for 4D input (B, C, H, W) takes (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
        # We want to pad H by 10 top, 10 bottom => 12+20 = 32
        y = F.pad(y, (0, 0, 10, 10)) 
        
        y = self.ecg_features(y)
        y = self.ecg_pool(y).flatten(1)          # -> (B, ecg_out_dim)
            
        return self.classifier(y)
