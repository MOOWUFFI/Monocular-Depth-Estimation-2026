"""TinyDepthUNet — MobileNetV3-Small + optional ASPP + multi-stride EAM + U-Net decoder.

Forward returns a dict:
    log_depth:  [B, 1, H, W]   predicted log-depth
    Us:         dict[int, tensor]   eigenbasis per active EAM stride
    eigvals:    dict[int, tensor]
    logits:     dict[int, tensor]   EAM pre-softmax cluster scores (feeds the losses)
    rgb_low:    [B, 3, h_min, w_min] RGB downsampled to the deepest EAM grid (for viz)

EAM scales are independent: stride 32 plugs in post-ASPP (channel
`bottleneck_channels`), stride 16 plugs in post-up1 (channel
`decoder_channels[0]`). Each scale instantiates its own
EigenAggregationModule sized to that channel count.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from approaches.Mobilenet.eagle import EigenAggregationModule


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _ASPPLite(nn.Module):
    """Tiny ASPP at the bottleneck: 1x1 + two dilated 3x3 + image pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.gap_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        gp = self.gap_conv(F.adaptive_avg_pool2d(x, 1))
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        cat = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), gp], dim=1)
        return self.project(cat)


class _UpBlock(nn.Module):
    """ConvTranspose 2x + concat skip + 3x3 conv + GN + ReLU + 3x3 conv + GN + ReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=False)
        in_after = in_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_after, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# Channels at each decoder/encoder stage — keys are spatial strides, values
# are channel counts. Drives EAM sizing per stride.
_STAGE_CHANNELS_AT_STRIDE = {
    32: "bottleneck",   # post-ASPP
    16: "decoder0",     # post-up1
    8:  "decoder1",     # post-up2
}


class TinyDepthUNet(nn.Module):
    """MobileNetV3-Small encoder + ASPP + U-Net decoder, with multi-stride EAGLE hooks.

    eam_scales: iterable of strides to plug EAM at. Supported: (), (32,),
        (32, 16). The encoder is stride-32 at its deepest; EAMs above that
        operate on decoder features.
    """

    def __init__(
        self,
        bottleneck_channels: int = 64,
        decoder_channels: tuple[int, int, int, int] = (64, 48, 32, 16),
        eam_scales: tuple[int, ...] = (),
        eam_k: int = 4,
        num_clusters: int = 10,
        eam_sigma_color: float = 0.2,
        pretrained_encoder: bool = True,
        use_aspp: bool = True,
    ) -> None:
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_encoder else None
        mbn = models.mobilenet_v3_small(weights=weights)

        # Patch stem from 3-ch RGB to 5-ch (RGB + X/Y coord grids).
        old_stem_conv = mbn.features[0][0]
        new_stem_conv = nn.Conv2d(
            5, old_stem_conv.out_channels,
            kernel_size=old_stem_conv.kernel_size,
            stride=old_stem_conv.stride,
            padding=old_stem_conv.padding,
            bias=(old_stem_conv.bias is not None),
        )
        with torch.no_grad():
            new_stem_conv.weight[:, :3, :, :] = old_stem_conv.weight
            new_stem_conv.weight[:, 3:, :, :] = 0.0
            if old_stem_conv.bias is not None:
                new_stem_conv.bias[:] = old_stem_conv.bias
        mbn.features[0][0] = new_stem_conv

        # Encoder stages, exposing skip features at each stride.
        self.enc_stem = mbn.features[0]                          # stride 2,  16 ch
        self.enc_s4 = mbn.features[1]                            # stride 4,  16 ch
        self.enc_s8 = nn.Sequential(*list(mbn.features[2:4]))    # stride 8,  24 ch
        self.enc_s16 = nn.Sequential(*list(mbn.features[4:9]))   # stride 16, 48 ch
        self.enc_s32 = nn.Sequential(*list(mbn.features[9:12]))  # stride 32, 96 ch

        # Bottleneck: ASPP-lite or plain 1×1 projection.
        self.use_aspp = bool(use_aspp)
        if self.use_aspp:
            self.bottleneck = _ASPPLite(96, bottleneck_channels)
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(96, bottleneck_channels, 1, bias=False),
                nn.GroupNorm(min(8, bottleneck_channels), bottleneck_channels),
                nn.ReLU(inplace=True),
            )

        # EAM hooks. Sized per-stride to the channel count at that stage.
        self.eam_scales = tuple(sorted(set(eam_scales), reverse=True))
        eam_channel = {
            32: bottleneck_channels,
            16: decoder_channels[0],
            8:  decoder_channels[1],
        }
        for s in self.eam_scales:
            if s not in eam_channel:
                raise ValueError(
                    f"eam_scales contains unsupported stride {s}; valid: 32, 16, 8."
                )
        self.eams = nn.ModuleDict({
            str(s): EigenAggregationModule(
                channels=eam_channel[s],
                k=eam_k,
                num_clusters=num_clusters,
                sigma_color=eam_sigma_color,
            )
            for s in self.eam_scales
        })

        # Decoder: 4 up-blocks (stride 32->16->8->4->2 + final upsample).
        c1, c2, c3, c4 = decoder_channels
        self.up1 = _UpBlock(bottleneck_channels, skip_ch=48, out_ch=c1)
        self.up2 = _UpBlock(c1, skip_ch=24, out_ch=c2)
        self.up3 = _UpBlock(c2, skip_ch=16, out_ch=c3)
        self.up4 = _UpBlock(c3, skip_ch=0, out_ch=c4)

        # Head -> log-depth. Small init so step 0 outputs ~log(1) = 0.
        self.head = nn.Conv2d(c4, 1, 1)
        nn.init.kaiming_normal_(self.head.weight, mode="fan_in", nonlinearity="linear")
        with torch.no_grad():
            self.head.weight.mul_(0.01)
        nn.init.zeros_(self.head.bias)

        # ImageNet stats for rgb_low denormalization (EAM colour affinity).
        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def _coord_channels(self, rgb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = rgb.shape
        y = torch.linspace(-1, 1, h, device=rgb.device, dtype=rgb.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        x = torch.linspace(-1, 1, w, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, y], dim=1)

    def _eam_step(self, stride: int, feat: torch.Tensor, rgb_denorm: torch.Tensor,
                  Us, eigvals, logits):
        if stride not in [int(s) for s in self.eam_scales]:
            return feat
        rgb_at = F.interpolate(rgb_denorm, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        feat, U, lam, lg = self.eams[str(stride)](feat, rgb_at)
        Us[stride] = U
        eigvals[stride] = lam
        logits[stride] = lg
        return feat

    def forward(self, rgb: torch.Tensor) -> dict:
        b, _, h, w = rgb.shape
        coords = self._coord_channels(rgb)
        x_in = torch.cat([rgb, coords], dim=1)

        x2 = self.enc_stem(x_in)
        s4 = self.enc_s4(x2)
        s8 = self.enc_s8(s4)
        s16 = self.enc_s16(s8)
        s32 = self.enc_s32(s16)

        bottleneck = self.bottleneck(s32)

        Us, eigvals, logits = {}, {}, {}
        rgb_denorm = (rgb * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)

        # EAM at stride 32 (post-ASPP).
        bottleneck = self._eam_step(32, bottleneck, rgb_denorm, Us, eigvals, logits)

        # Decoder. EAM at stride 16 hooks in post-up1; at stride 8 post-up2.
        d1 = self.up1(bottleneck, s16)
        d1 = self._eam_step(16, d1, rgb_denorm, Us, eigvals, logits)
        d2 = self.up2(d1, s8)
        d2 = self._eam_step(8, d2, rgb_denorm, Us, eigvals, logits)
        d3 = self.up3(d2, s4)
        d4 = self.up4(d3, None)
        log_depth = self.head(d4)
        log_depth = F.interpolate(log_depth, size=(h, w), mode="bilinear", align_corners=False)

        # RGB at the deepest EAM grid (stride 32) for downstream viz.
        rgb_low = F.interpolate(rgb_denorm, size=bottleneck.shape[-2:],
                                mode="bilinear", align_corners=False)

        return {
            "log_depth": log_depth,
            "Us": Us,
            "eigvals": eigvals,
            "logits": logits,
            "rgb_low": rgb_low,
        }

    def param_groups(self, base_lr: float, encoder_mult: float, weight_decay: float):
        """Encoder stages at `encoder_mult * lr`, rest at `lr`. No WD on norms / biases."""
        encoder_modules = (self.enc_stem, self.enc_s4, self.enc_s8, self.enc_s16, self.enc_s32)
        encoder_param_ids = {id(p) for m in encoder_modules for p in m.parameters()}
        enc_decay, enc_nodecay, oth_decay, oth_nodecay = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_encoder = id(p) in encoder_param_ids
            no_decay = p.ndim == 1 or name.endswith(".bias")
            if is_encoder and no_decay:
                enc_nodecay.append(p)
            elif is_encoder:
                enc_decay.append(p)
            elif no_decay:
                oth_nodecay.append(p)
            else:
                oth_decay.append(p)
        return [
            {"params": enc_decay,   "lr": base_lr * encoder_mult, "weight_decay": weight_decay},
            {"params": enc_nodecay, "lr": base_lr * encoder_mult, "weight_decay": 0.0},
            {"params": oth_decay,   "lr": base_lr,                "weight_decay": weight_decay},
            {"params": oth_nodecay, "lr": base_lr,                "weight_decay": 0.0},
        ]


def build_model(args_dict: dict, device) -> "TinyDepthUNet":
    """Construct a TinyDepthUNet from a checkpoint's saved ``args`` dict (or a
    training argparse namespace converted to a dict). Used by both training and
    inference so the architecture is reconstructed identically."""
    return TinyDepthUNet(
        bottleneck_channels=int(args_dict.get("bottleneck_channels", 64)),
        decoder_channels=tuple(args_dict.get("decoder_channels", [64, 48, 32, 16])),
        eam_scales=tuple(args_dict.get("eam_scales", [])),
        eam_k=int(args_dict.get("eam_k", 4)),
        num_clusters=int(args_dict.get("num_clusters", 10)),
        eam_sigma_color=float(args_dict.get("eam_sigma_color", 0.2)),
        pretrained_encoder=bool(args_dict.get("pretrained_encoder", False)),
        use_aspp=bool(args_dict.get("use_aspp", True)),
    ).to(device)
