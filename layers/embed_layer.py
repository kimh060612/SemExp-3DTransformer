import clip
import torch
from torch import nn as nn
from torch.nn import functional as F
from gym import spaces
from torchvision import transforms as T
from torchvision.transforms import functional as TF

class ResNetCLIPEncoder(nn.Module):
    def __init__(self, pooling='attnpool', device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.rgb = True
        self.depth = True

        if not self.is_blind:
            model, preprocess = clip.load("RN50", device=device, jit=False)
            model = model.float()
            # expected input: C x H x W (np.uint8 in [0-255])
            self.preprocess = T.Compose([
                # resize and center crop to 224
                preprocess.transforms[0],  
                preprocess.transforms[1],
                # already tensor, but want float
                T.ConvertImageDtype(torch.float),
                # normalize with CLIP mean, std
                preprocess.transforms[4],
            ])
            # expected output: C x H x W (np.float32)

            self.backbone = model.visual

            if self.rgb and self.depth:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048,)
            elif pooling == 'none':
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif pooling == 'avgpool':
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1,1)),
                    nn.Flatten()
                )
                self.output_shape = (2048,)
            else:
                self.output_shape = (1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self.rgb is False and self.depth is False

    def forward(self, observations) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None
        ### observation: BATCH * CHANNEL * HEIGHT * WIDTH
        cnn_input = []
        if self.rgb:
            rgb_observations = observations[:, :3, ...]
            rgb_observations = torch.stack(
                [ self.preprocess(rgb_image) for rgb_image in rgb_observations ]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            rgb_x = self.backbone(rgb_observations).float()
            cnn_input.append(rgb_x)

        if self.depth:
            depth_observations = observations[:, 0, ...]  # [BATCH x HEIGHT X WIDTH]
            ddd = torch.stack([depth_observations] * 3, dim=1)  # [BATCH x 3 x HEIGHT X WIDTH]
            ddd = torch.stack([
                self.preprocess(TF.convert_image_dtype(depth_map, torch.uint8))
                for depth_map in ddd
            ])  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            depth_x = self.backbone(ddd).float()
            cnn_input.append(depth_x)

        if self.rgb and self.depth:
            x = F.adaptive_avg_pool2d(cnn_input[0] + cnn_input[1], 1)
            x = x.flatten(1)
        else:
            x = torch.cat(cnn_input, dim=1)

        return x