from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet-18/34 使用的 basic residual block.

    结构：Conv3x3-BN-ReLU-Conv3x3-BN-Add-ReLU。
    当 stride != 1 或通道数变化时，使用 1x1 projection shortcut。
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        use_shortcut: bool = True,
        shortcut_type: str = "projection",
    ) -> None:
        super().__init__()
        self.use_shortcut = use_shortcut
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module
        if not use_shortcut:
            self.shortcut = nn.Identity()
        elif stride != 1 or in_planes != planes * self.expansion:
            if shortcut_type == "projection":
                self.shortcut = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride),
                    nn.BatchNorm2d(planes * self.expansion),
                )
            elif shortcut_type == "identity_pad":
                self.shortcut = IdentityPad(in_planes, planes * self.expansion, stride)
            else:
                raise ValueError(f"Unknown shortcut_type: {shortcut_type}")
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_shortcut:
            out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class PlainBlock(nn.Module):
    """PlainNet 对照组 block：与 BasicBlock 参数规模接近，但没有 shortcut add。"""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, **_: object) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # 没有 residual add，但为了保证下一个 block 的输入通道数正确，直接输出 planes 通道。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class IdentityPad(nn.Module):
    """ResNet 论文 option A：stride 下采样 + channel zero padding，不引入额外参数。"""

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.out_planes > self.in_planes:
            pad_channels = self.out_planes - self.in_planes
            zeros = torch.zeros(
                x.size(0), pad_channels, x.size(2), x.size(3),
                dtype=x.dtype, device=x.device,
            )
            x = torch.cat([x, zeros], dim=1)
        return x


_DEPTH_TO_LAYERS = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
}


class ResNet(nn.Module):
    def __init__(
        self,
        depth: int = 18,
        num_classes: int = 10,
        in_channels: int = 3,
        block: type[nn.Module] = BasicBlock,
        use_shortcut: bool = True,
        shortcut_type: str = "projection",
        zero_init_residual: bool = False,
        cifar_stem: bool = True,
    ) -> None:
        super().__init__()
        if depth not in _DEPTH_TO_LAYERS:
            raise ValueError(f"Only depths {list(_DEPTH_TO_LAYERS)} are implemented for BasicBlock.")
        layers = _DEPTH_TO_LAYERS[depth]
        self.in_planes = 64
        self.cifar_stem = cifar_stem

        if cifar_stem:
            self.conv1 = conv3x3(in_channels, 64, stride=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, use_shortcut=use_shortcut, shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_shortcut=use_shortcut, shortcut_type=shortcut_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_shortcut=use_shortcut, shortcut_type=shortcut_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_shortcut=use_shortcut, shortcut_type=shortcut_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual=zero_init_residual)

    def _make_layer(
        self,
        block: type[nn.Module],
        planes: int,
        blocks: int,
        stride: int,
        use_shortcut: bool,
        shortcut_type: str,
    ) -> nn.Sequential:
        layers = []
        layers.append(block(self.in_planes, planes, stride, use_shortcut=use_shortcut, shortcut_type=shortcut_type))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, use_shortcut=use_shortcut, shortcut_type=shortcut_type))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool = False) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PlainNet(ResNet):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["block"] = PlainBlock
        kwargs["use_shortcut"] = False
        super().__init__(*args, **kwargs)


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    name = model_cfg.get("name", "resnet").lower()
    depth = int(model_cfg.get("depth", 18))
    num_classes = int(dataset_cfg.get("num_classes", 10))
    in_channels = int(dataset_cfg.get("in_channels", 3))
    common_kwargs = dict(
        depth=depth,
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_type=model_cfg.get("shortcut_type", "projection"),
        zero_init_residual=bool(model_cfg.get("zero_init_residual", False)),
        cifar_stem=bool(model_cfg.get("cifar_stem", True)),
    )
    if name == "resnet":
        return ResNet(
            block=BasicBlock,
            use_shortcut=bool(model_cfg.get("use_shortcut", True)),
            **common_kwargs,
        )
    if name == "plainnet":
        return PlainNet(**common_kwargs)
    raise ValueError(f"Unknown model name: {name}")
