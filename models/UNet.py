import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        """
        The double convolution layer used per block in UNet Architecture
        Args:
            in_channels: # of channels in input image. E.g 3 channels for RGB images.
            out_channels: # of filters applied to input image. Each filter detects different features from input
        """
        super(DoubleConv, self).__init__()
        # Sequential module to map out the double conv layer per block in UNET Architecture
        self.double_conv = nn.Sequential(
            # Kernel size of 3, stride 1, and padding 1 will help achieve same convolution
            # Bias is set to false since BatchNorm will just cancel it out
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # ReLU activation will apply non-linearity after convolution and maintain it during normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Args:
        in_channels: Begin with 3 for an RGB image.
        out_channels: Begin with 5 for the categories: shirts, pants, dresses, accessories, shoes
        features: Feature mapping for downsampling and upsampling in the paths of the UNet architecture
    """
    def __init__(self, in_channels = 3, out_channels = 5, features = [64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Pooling layer used for down sampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path (encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling path (decoder)
        for feature in reversed(features):
            # Use bilinear upsampling and a conv layer
            self.ups.append(
                nn.Sequential(
                    # Double the size of input feature map
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1)
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
