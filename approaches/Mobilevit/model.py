"""MobileViT-XX-Small encoder + DepthDecodeHead + EAGLE EigenAggregationModules.

Specialised to the MobileViT-XX-Small backbone (``apple/mobilevit-xx-small``,
5 encoder stages, ``hidden_sizes = neck_hidden_sizes[1:-1] = [32, 64, 96, 128, 160]``).
The EAMs are *side-channel* modules: they do NOT modify the feature maps fed to
the depth decoder; their outputs (``U``, ``eigvals``, ``logits``) feed the
auxiliary EAGLE losses only.

MobileViT input is raw [0, 1] (no ImageNet shift) — ``input_is_imagenet_normalized``
defaults to False.

Forward returns a dict:
    depth   : (B, H, W)             predicted metric depth
    logits  : dict[int, (B, N, K)]  EAM pre-softmax cluster scores (feeds losses)
    Us      : dict[int, (B, N, k)]  eigenvectors (for visualisation)
    eigvals : dict[int, (B, k)]     eigenvalues (for visualisation)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MobileViTConfig, MobileViTModel

from approaches.Mobilevit.eagle import EigenAggregationModule

# MobileViT-XX-Small hidden sizes per encoder stage (256x256 input).
# neck_hidden_sizes[1:-1] from apple/mobilevit-xx-small config.
MOBILEVIT_XXS_HIDDEN_SIZES = [32, 64, 96, 128, 160]


# ── DepthDecodeHead (SEQUENTIAL concat order) ───────────────────────────────

class DepthDecodeHead(nn.Module):
    """All-MLP depth decode head (SegFormer-style, single-channel metric depth).

    For each encoder stage i:
      (B, C_i, H_i, W_i) -> Linear(C_i -> D) -> bilinear upsample to stage-0 res

    Concatenate all stages in sequential order (stage 0 first) ->
    fuse Conv(num_stages*D -> D) + BN + ReLU + Dropout -> depth Conv(D -> 1)
    -> bilinear upsample to input res -> Softplus (positive depth).
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_stages = len(hidden_sizes)
        self.linear_c = nn.ModuleList(
            [nn.Linear(c, decoder_hidden_size) for c in hidden_sizes]
        )
        self.linear_fuse = nn.Conv2d(
            self.num_stages * decoder_hidden_size,
            decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.activation = nn.ReLU(inplace=True)
        self.dropout    = nn.Dropout(p=dropout)
        self.depth_head = nn.Conv2d(decoder_hidden_size, 1, kernel_size=1)

    def forward(
        self,
        encoder_hidden_states: Tuple[torch.Tensor, ...],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        ref_size = encoder_hidden_states[0].shape[-2:]
        upsampled: List[torch.Tensor] = []
        for x, linear in zip(encoder_hidden_states, self.linear_c):
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            x_proj = linear(x_flat)
            x_proj = (
                x_proj.reshape(B, H, W, -1)
                       .permute(0, 3, 1, 2)
                       .contiguous()
            )
            if (H, W) != ref_size:
                x_proj = F.interpolate(
                    x_proj, size=ref_size, mode="bilinear", align_corners=False
                )
            upsampled.append(x_proj)

        x = torch.cat(upsampled, dim=1)  # sequential order: stage 0 first
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.depth_head(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return F.softplus(x).squeeze(1)  # (B, H, W)


# ── MobileViT + EAGLE model ─────────────────────────────────────────────────

class EAGLEDepthModel(nn.Module):
    """MobileViT encoder + DepthDecodeHead + EigenAggregationModules (EAMs).

    The EAMs are *side-channel* modules — they do NOT modify the feature maps
    fed to the depth decoder. For each requested encoder stage s the EAM:

      1. Builds a patch-level affinity graph A = A_color + A_seg.
      2. Computes the symmetric normalized graph Laplacian L_sym.
      3. Takes the k smallest non-trivial eigenvectors of L_sym.
      4. Computes soft cluster assignment logits via learnable cluster centres.

    Multi-layer affinity (``multi_layer_affinity``) builds the semantic
    affinity from concatenated features of the three preceding stages,
    projected to the stage-s channel dim.

    MobileViT-XXS spatial geometry (256x256 input, divide by 2x2^s):
      Stage 0: 32 ch,  128x128
      Stage 1: 64 ch,  64x64
      Stage 2: 96 ch,  32x32
      Stage 3: 128 ch, 16x16 -> N=256
      Stage 4: 160 ch, 8x8   -> N=64
    Use stages 3 / 4 for the EAMs (small grids); earlier stages have too many
    patches for the N x N affinity.
    """

    def __init__(
        self,
        encoder: "MobileViTModel",
        depth_head: DepthDecodeHead,
        hidden_sizes: List[int],
        eam_stages: List[int],
        eam_k: int = 4,
        num_clusters: int = 10,
        eam_sigma_color: float = 0.2,
        multi_layer_affinity: bool = False,
        differentiable_eigh: bool = False,
        eigh_eps_reg: float = 1e-2,
        input_is_imagenet_normalized: bool = False,
    ) -> None:
        super().__init__()
        self.encoder      = encoder
        self.depth_head   = depth_head
        self.eam_stages   = sorted(eam_stages)
        self.hidden_sizes = hidden_sizes
        self.multi_layer_affinity = multi_layer_affinity
        self.input_is_imagenet_normalized = input_is_imagenet_normalized

        # EAM for each requested stage
        self.eams = nn.ModuleDict({
            str(s): EigenAggregationModule(
                channels=hidden_sizes[s],
                k=eam_k,
                num_clusters=num_clusters,
                sigma_color=eam_sigma_color,
                use_residual=False,
                differentiable_eigh=differentiable_eigh,
                eigh_eps_reg=eigh_eps_reg,
            )
            for s in self.eam_stages
        })

        # Optional multi-layer affinity projections.
        if multi_layer_affinity:
            self.affinity_projs = nn.ModuleDict()
            for s in self.eam_stages:
                src_stages = [max(0, s - 2), max(0, s - 1), s]
                seen: list[int] = []
                for si in src_stages:
                    if si not in seen:
                        seen.append(si)
                concat_ch = sum(hidden_sizes[si] for si in seen)
                if concat_ch != hidden_sizes[s]:
                    self.affinity_projs[str(s)] = nn.Conv2d(
                        concat_ch, hidden_sizes[s], 1, bias=False
                    )

        # ImageNet unnormalization buffers (for color affinity)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _build_affinity_feat(
        self,
        s: int,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Optional[torch.Tensor]:
        """Build the affinity feature tensor for stage s.

        Single-layer (default): returns None -> EAM uses feat directly.
        Multi-layer: concatenates the 3 preceding stages (or fewer at the
        boundaries), projects to stage-s channel dim, and returns the result.
        """
        if not self.multi_layer_affinity:
            return None

        src_stages: list[int] = []
        for si in [max(0, s - 2), max(0, s - 1), s]:
            if si not in src_stages:
                src_stages.append(si)

        target_size = hidden_states[s].shape[-2:]
        parts = []
        for si in src_stages:
            hi = hidden_states[si]
            if hi.shape[-2:] != target_size:
                hi = F.interpolate(hi, size=target_size, mode="bilinear", align_corners=False)
            parts.append(hi)

        concat = torch.cat(parts, dim=1)

        proj_key = str(s)
        if proj_key in self.affinity_projs:
            return self.affinity_projs[proj_key](concat)
        return concat

    def forward(self, pixel_values: torch.Tensor) -> Dict:
        """
        Args:
            pixel_values: (B, 3, H, W) — raw [0, 1] for MobileViT.

        Returns a dict with keys depth / logits / Us / eigvals.
        """
        # 1. Encoder — collect all hidden states
        enc_out      = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = enc_out.hidden_states   # 5 x (B, C_i, H_i, W_i)
        H, W          = pixel_values.shape[-2:]

        # 2. Recover raw [0, 1] RGB for the color affinity.
        if self.input_is_imagenet_normalized:
            rgb_raw = (
                pixel_values * self.imagenet_std + self.imagenet_mean
            ).clamp(0.0, 1.0)
        else:
            rgb_raw = pixel_values.clamp(0.0, 1.0)

        # 3. Run EAMs at requested stages (side-channel only)
        logits_dict: Dict[int, torch.Tensor]  = {}
        Us_dict:     Dict[int, torch.Tensor]  = {}
        eigvals_dict: Dict[int, torch.Tensor] = {}

        for s in self.eam_stages:
            feat   = hidden_states[s]
            rgb_lo = F.interpolate(
                rgb_raw, size=feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            affinity_feat = self._build_affinity_feat(s, hidden_states)
            _, U, lam, lg = self.eams[str(s)](feat, rgb_lo, affinity_feat=affinity_feat)
            logits_dict[s]  = lg
            Us_dict[s]      = U
            eigvals_dict[s] = lam

        # 4. Depth prediction (encoder hidden states UNCHANGED by EAMs)
        depth = self.depth_head(hidden_states, target_size=(H, W))  # (B, H, W)

        return {
            "depth":   depth,
            "logits":  logits_dict,
            "Us":      Us_dict,
            "eigvals": eigvals_dict,
        }

    def param_groups(
        self,
        base_lr: float,
        encoder_lr_mult: float,
        weight_decay: float,
    ) -> List[dict]:
        """AdamW param groups.

        - encoder params:    encoder_lr_mult x base_lr
        - decoder + EAM:     base_lr
        No weight decay on 1-D params (biases, BN/LN weights).
        """
        enc_wd, enc_nd, oth_wd, oth_nd = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_enc = name.startswith("encoder.")
            no_wd  = p.ndim == 1 or name.endswith(".bias")
            if is_enc and no_wd:
                enc_nd.append(p)
            elif is_enc:
                enc_wd.append(p)
            elif no_wd:
                oth_nd.append(p)
            else:
                oth_wd.append(p)
        return [
            {"params": enc_wd,  "lr": base_lr * encoder_lr_mult, "weight_decay": weight_decay},
            {"params": enc_nd,  "lr": base_lr * encoder_lr_mult, "weight_decay": 0.0},
            {"params": oth_wd,  "lr": base_lr,                   "weight_decay": weight_decay},
            {"params": oth_nd,  "lr": base_lr,                   "weight_decay": 0.0},
        ]


def build_encoder(args_dict: dict, device):
    """Build the MobileViT encoder from a config dict and return
    (encoder, hidden_sizes, input_is_imagenet_normalized).

    Weights come from the pretrained / stage-2 checkpoint. ``stage2_encoder``
    (a HuggingFace MobileViTModel save directory), when provided and present,
    is loaded directly (config + weights) to sidestep a HF repo-ID validation
    quirk on local paths.
    """
    mobilevit_model_id = args_dict.get("mobilevit_model_id", "apple/mobilevit-xx-small")
    stage2_encoder = args_dict.get("stage2_encoder", "") or ""

    if stage2_encoder:
        from pathlib import Path
        encoder_path = Path(stage2_encoder).resolve()
        mvit_config = MobileViTConfig.from_pretrained(str(encoder_path))
        encoder = MobileViTModel(mvit_config)
        sf_path  = encoder_path / "model.safetensors"
        bin_path = encoder_path / "pytorch_model.bin"
        if sf_path.exists():
            try:
                from safetensors.torch import load_file as _sf_load
                _sd = _sf_load(str(sf_path))
            except ImportError:
                _sd = torch.load(str(sf_path), map_location="cpu", weights_only=True)
        elif bin_path.exists():
            _sd = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No weights file (model.safetensors / pytorch_model.bin) found in {encoder_path}"
            )
        encoder.load_state_dict(_sd, strict=False)
    else:
        encoder = MobileViTModel.from_pretrained(mobilevit_model_id)

    # neck_hidden_sizes[1:-1] are the per-encoder-layer output channels.
    hidden_sizes = list(encoder.config.neck_hidden_sizes[1:-1])
    return encoder.to(device), hidden_sizes, False


def build_model(args_dict: dict, device, encoder=None, hidden_sizes=None):
    """Construct an ``EAGLEDepthModel`` from a saved ``args`` dict.

    If ``encoder`` / ``hidden_sizes`` are not supplied they are built from the
    config in ``args_dict`` via :func:`build_encoder`.
    """
    if encoder is None or hidden_sizes is None:
        encoder, hidden_sizes, _ = build_encoder(args_dict, device)

    depth_head = DepthDecodeHead(
        hidden_sizes=hidden_sizes,
        decoder_hidden_size=int(args_dict.get("decoder_hidden_size", 256)),
        dropout=float(args_dict.get("decoder_dropout", 0.1)),
    ).to(device)

    model = EAGLEDepthModel(
        encoder=encoder,
        depth_head=depth_head,
        hidden_sizes=hidden_sizes,
        eam_stages=list(args_dict.get("eam_stages", [3, 4])),
        eam_k=int(args_dict.get("eam_k", 4)),
        num_clusters=int(args_dict.get("num_clusters", 10)),
        eam_sigma_color=float(args_dict.get("eam_sigma_color", 0.2)),
        multi_layer_affinity=bool(args_dict.get("multi_layer_affinity", False)),
        differentiable_eigh=bool(args_dict.get("differentiable_eigh", False)),
        input_is_imagenet_normalized=False,
    ).to(device)
    return model
