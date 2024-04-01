import torch
import torch.nn.functional as F
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) # /stride
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # inplace 옵션의 역할이 무엇인가?
        self.conv2 =  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # 
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.im = lambda x: x # identity mapping
        
    def forward(self, x):
        residual=self.im(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

### TODO: 3D convolutional layer for K*M*M egocentric Map, Simple ResNet Architecture
class Conv3dMap(nn.Module):
    def __init__(self, input_shape, emb_3d_vecsz, num_sem_categories=16, hidden_size=512):
        super(Conv3dMap, self).__init__()
        
        self.input_dims = 64
        self.output_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)
        self.premp1 = nn.MaxPool2d(2) # Tensor Size(wo. Batch): H*W -> (H//2)*(W//2)
        self.pre_conv1 = nn.Conv2d(num_sem_categories + 8, 64, 3, stride=1, padding=1) # PADDING SAME for kernel=3, stride=1, padding=1
        self.pre_bn1 = nn.BatchNorm2d(64)
        self.pre_relu = nn.ReLU(inplace=True)
        self.layer1 = self._build_layer(2, 64) # Preserve Tensor Size
        self.layer2 = self._build_layer(2, 128, stride=2) # Downsample to Half
        self.layer3 = self._build_layer(2, 256, stride=2) # Downsample to Half
        self.layer4 = self._build_layer(2, 512, stride=2) # Downsample to Half
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.output_size * 32 + 8 * 2)
        
        self.linear1 = nn.Linear(self.output_size * 32 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, emb_3d_vecsz)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _build_layer(self, num_layers, out_dims, stride=1):
        downsample = None
        if stride != 1: 
            downsample = nn.Sequential(
                nn.Conv2d(self.input_dims, out_dims, 1, stride),
                nn.BatchNorm2d(out_dims)
            )
        layers = []
        layers.append(ResidualBlock(self.input_dims, out_dims, stride, downsample))
        self.input_dims = out_dims
        
        for _ in range(1, num_layers): 
            layers.append(ResidualBlock(out_dims, out_dims))
        return nn.Sequential(*layers)
    
    def forward_emb(self, x):
        x = self.premp1(x)
        x = self.pre_conv1(x)
        x = self.pre_bn1(x)
        x = self.pre_relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ap(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.ReLU()(x)
        
    def forward(self, x):
        x = self.forward_emb(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x