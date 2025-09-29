import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# choose backbone from above imports :)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module 
    """
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        self.project = nn.Sequential(
            nn.Conv2d((len(atrous_rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        features = []
        
        # Image-level features via global average pooling
        feat1 = self.global_avg_pool(x)
        feat1 = F.interpolate(feat1, size=x.size()[2:], mode='bilinear', align_corners=True)
        features.append(feat1)
        
        # 1x1 convolution
        feat2 = self.conv1(x)
        features.append(feat2)
        
        # Atrous convolutions
        for atrous_conv in self.atrous_convs:
            feat = atrous_conv(x)
            features.append(feat)
        
        # Concatenate and project
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

class Decoder(nn.Module):
    """
    This upsamples the ASPP output and combines it with low-level features from the backbone.
    """
    def __init__(self, low_level_channels, aspp_channels, out_channels=256, classes=1, image_size=(512, 512)):
        super(Decoder, self).__init__()
        
        # Store image size for dynamic upsampling
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.final_conv = nn.Conv2d(out_channels, classes, 1)

    def forward(self, aspp_out, low_level_feat):
        # Upsample ASPP output to match low-level feature size
        aspp_out = F.interpolate(aspp_out, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # Concatenate and decode
        x = torch.cat([aspp_out, low_level_feat], dim=1)
        x = self.decoder_conv(x)
        
        # Final output, upsample to original input size based on image_size
        x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        return x

class DeepLabV3Plus(nn.Module):
    """    
    Key components:
    - Backbone: ResNet50 (or other variants) for feature extraction.
    - ASPP: Captures multi-scale information with atrous convolutions.
    - Decoder: Fuses high-level and low-level features for refined segmentation.
    
    Notes:
    - This implementation assumes the backbone outputs features at 1/4 (low-level) and 1/32 (high-level) of input resolution.
    """
    def __init__(self, 
                 encoder_name="resnet50", 
                 encoder_weights="imagenet",
                 in_channels=3,
                 classes=1,
                 activation=None,
                 pretrained=True,
                 image_size=(512, 512)):
        super(DeepLabV3Plus, self).__init__()
        
        self.encoder_name = encoder_name
        # calculate every stage feature map sizes
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.low_level_size = (self.image_size[0] // 4, self.image_size[1] // 4)  # 1/4 resolution
        self.high_level_size = (self.image_size[0] // 32, self.image_size[1] // 32)  # 1/32 resolution
        
        # Load backbone based on encoder
        if encoder_name == "resnet50":
            if pretrained and encoder_weights == "imagenet":
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet50(weights=None)
            
            # Modify first conv if in_channels != 3
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            self.low_level_channels = 256   # After layer1 (res2)
            self.high_level_channels = 2048 # After layer4 (res5)
            
        elif encoder_name == "resnet34":
            if pretrained and encoder_weights == "imagenet":
                self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet34(weights=None)
            
            # Modify first conv if in_channels != 3
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            self.low_level_channels = 64    # After layer1 (res2)
            self.high_level_channels = 512  # After layer4 (res5)

        elif encoder_name == "mobilenet_v2":
            if pretrained and encoder_weights == "imagenet":
                self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                self.backbone = mobilenet_v2(weights=None)

            # Modify first conv if in_channels != 3
            if in_channels != 3:
                # MobileNetV2 uses features[0][0] as first conv
                self.backbone.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
            # MobileNetV2 channel dimensions
            self.low_level_channels = 24    # After early feature layers
            self.high_level_channels = 1280 # After final feature layers
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}. Supported encoders: 'resnet50', 'resnet34', 'mobilenet_v2'.")
        
        if encoder_name in ["resnet50", "resnet34"]:
            self.layer0 = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool
            )
            self.layer1 = self.backbone.layer1  # Low-level features
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4  # High-level features
        elif encoder_name == "mobilenet_v2":
            # MobileNetV2 uses sequential features
            self.features = self.backbone.features
        
        # ASPP module
        self.aspp = ASPP(in_channels=self.high_level_channels, out_channels=256)
        
        # Decoder module
        self.decoder = Decoder(low_level_channels=self.low_level_channels, aspp_channels=256, out_channels=256, classes=classes, image_size=self.image_size)
        
        self.activation = activation  # Store for forward if needed

    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        
        # Backbone forward
        if self.encoder_name in ["resnet50", "resnet34"]:
            x0 = self.layer0(x)
            low_level = self.layer1(x0)  # Low-level features (1/4 resolution)
            x2 = self.layer2(low_level)
            x3 = self.layer3(x2)
            high_level = self.layer4(x3)  # High-level features (1/32 resolution)
        elif self.encoder_name == "mobilenet_v2":
            # MobileNetV2 forward pass
            low_level = None
            high_level = x
            
            for i, layer in enumerate(self.features):
                high_level = layer(high_level)
                # Extract low-level features after early layers (stage 3-4)
                if i == 3 and low_level is None:  # After some initial layers
                    low_level = high_level
        
        # ASPP
        aspp_out = self.aspp(high_level)
        
        # Update decoder's target size to match input
        self.decoder.image_size = input_size
        
        # Decoder
        out = self.decoder(aspp_out, low_level)
        out = torch.sigmoid(out)
        
        return out