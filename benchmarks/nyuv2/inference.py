"""NYUv2 cross-dataset depth benchmark.

Evaluates trained depth checkpoints on the labelled NYU-Depth-v2 test split
(Eigen indices + Eigen crop). For each model in the registry it reports SILog,
AbsRel, and delta1 after the standard least-squares affine alignment.

This script is self-contained: every model architecture is defined inline so it
can load any of the project's checkpoints (ResNet34+EAGLE, SegFormer, MobileViT,
MobileNet, and the standalone UNet/ResNet baselines) without importing the
training packages.

Data + checkpoint locations default to scripts.constants.BENCHMARK_DATA_DIR and
are overridable on the command line:

    python -m benchmarks.nyuv2.inference \\
        --data_dir   data/benchmarks/nyuv2 \\
        --models_dir data/benchmarks
"""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

from scripts.constants import BENCHMARK_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

NYU_MIN_DEPTH, NYU_MAX_DEPTH = 1e-3, 10.0
_EIGEN_CROP = (16, 464, 18, 622)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Benchmark data lives under BENCHMARK_DATA_DIR/nyuv2 by default; the trained
# checkpoints referenced by the registry below live under BENCHMARK_DATA_DIR.
# Both roots are overridable on the command line (see __main__).
_NYU_DIR    = os.path.join(BENCHMARK_DATA_DIR, "nyuv2")
MAT_PATH    = os.path.join(_NYU_DIR, "nyu_depth_v2_labeled.mat")
SPLITS_PATH = os.path.join(_NYU_DIR, "splits.mat")
MODELS_DIR  = BENCHMARK_DATA_DIR
USE_SPLIT   = True
USE_CROP    = True
INPUT_SIZE  = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL REGISTRY ──────────────────────────────────────────────────────────
# Add new models here. Each entry: (name, ckpt_path, arch_type, input_norm)
# arch_type: "resnet34_eagle" | "segformer" | "mobilevit" | "mobilenet" | "simple"
# input_norm: "imagenet" | "raw"

MODELS = [
    ("resnet34_pretrained","models/resnet34/pretrained.pt",   "resnet34_eagle", "imagenet"),
    ("resnet34_aspp",      "models/resnet34/aspp.pt",         "resnet34_eagle", "imagenet"),
    ("resnet34_aspp_eagle","models/resnet34/aspp_eagle.pt",   "resnet34_eagle", "imagenet"),
    ("mobilevit_scratch",  "models/mobilevit/scratch.pt",     "mobilevit",      "raw"),
    ("mobilevit_pretrained","models/mobilevit/pretrained.pt", "mobilevit",      "raw"),
    ("mobilevit_eagle",    "models/mobilevit/eagle.pt",       "mobilevit",      "raw"),
    ("mobilenet_scratch",  "models/mobilenet/scratch.pth",    "mobilenet",      "imagenet"),
    ("mobilenet_pretrained","models/mobilenet/pretrained.pth","mobilenet",      "imagenet"),
    ("segformer_scratch",  "models/segformer/scratch.pt",     "segformer",      "imagenet"),
    ("segformer_pretrained","models/segformer/pretrained.pt", "segformer",      "imagenet"),
    ("segformer_eagle",    "models/segformer/eagle.pt",       "segformer",      "imagenet"),
    ("unet_scratch",       "models/other_models/unet_scratch.pth",     "simple", "raw"),
    ("resnet18_scratch",   "models/other_models/resnet18_scratch.pth", "simple", "raw"),
    ("resnet34_scratch",   "models/resnet34/scratch.pth",              "simple", "raw"),
    ("resnet50_scratch",   "models/other_models/resnet50_scratch.pth", "simple", "raw"),
]

# ─── SHARED UTILITIES ────────────────────────────────────────────────────────

def open_mat(mat_path):
    f = h5py.File(mat_path, "r")
    return f, f["images"], f["depths"], f["images"].shape[0]

def read_sample(images, depths, idx):
    return images[idx].transpose(2, 1, 0).astype(np.uint8), depths[idx].T.astype(np.float32)

def load_eigen_indices(splits_path, n_total):
    raw = None
    try:
        import scipy.io
        mat = scipy.io.loadmat(splits_path)
        for key in ("testNdxs", "test_idx", "testIdx", "test"):
            if key in mat: raw = mat[key].flatten().astype(int); break
    except Exception: pass
    if raw is None:
        with h5py.File(splits_path, "r") as f:
            for key in ("testNdxs", "test_idx", "testIdx", "test"):
                if key in f: raw = f[key][()].flatten().astype(int); break
    indices = sorted(set(int(i) - 1 for i in raw))
    return [i for i in indices if 0 <= i < n_total]

def eigen_crop(arr):
    t, b, l, r = _EIGEN_CROP; return arr[t:b, l:r]

def preprocess(rgb, input_size, norm):
    rgb_r = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb_f = rgb_r.astype(np.float32) / 255.0
    if norm == "imagenet": rgb_f = (rgb_f - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(rgb_f.transpose(2, 0, 1)).unsqueeze(0)

def compute_metrics(pred, gt):
    valid = (gt > NYU_MIN_DEPTH) & (gt <= NYU_MAX_DEPTH) & (pred > NYU_MIN_DEPTH) & (pred <= NYU_MAX_DEPTH) & np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() < 10: return None
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]
    pa = np.clip(s * p + t, 1e-6, None)
    ld = np.log(pa) - np.log(g)
    return {"silog": float(np.sqrt(np.mean((ld - ld.mean()) ** 2) + 1e-10)), "abs_rel": float(np.mean(np.abs(pa - g) / g)), "delta1": float(np.mean(np.maximum(pa / g, g / pa) < 1.25))}

# ─── ARCH: SIMPLE (UNet / ResNet18/34/50 standalone) ─────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, i, o, d=0.0):
        super().__init__(); layers = [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)];
        if d > 0: layers.append(nn.Dropout2d(d))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__(); self.enc1=ConvBlock(3,64); self.enc2=ConvBlock(64,128); self.enc3=ConvBlock(128,256); self.enc4=ConvBlock(256,512); self.pool=nn.MaxPool2d(2); self.bottleneck=ConvBlock(512,512,0.3); self.up4=nn.ConvTranspose2d(512,512,2,stride=2); self.dec4=ConvBlock(1024,512); self.up3=nn.ConvTranspose2d(512,256,2,stride=2); self.dec3=ConvBlock(512,256); self.up2=nn.ConvTranspose2d(256,128,2,stride=2); self.dec2=ConvBlock(256,128); self.up1=nn.ConvTranspose2d(128,64,2,stride=2); self.dec1=ConvBlock(128,64); self.out=nn.Conv2d(64,1,1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3)); b=self.bottleneck(self.pool(e4)); d4=self.dec4(torch.cat([self.up4(b),e4],1)); d3=self.dec3(torch.cat([self.up3(d4),e3],1)); d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1)); return F.softplus(self.out(d1))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, i, o, s=1):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(i,o,3,stride=s,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o)); self.shortcut=(nn.Sequential(nn.Conv2d(i,o,1,stride=s,bias=False),nn.BatchNorm2d(o)) if s!=1 or i!=o else nn.Identity()); self.relu=nn.ReLU(True)
    def forward(self, x): return self.relu(self.net(x)+self.shortcut(x))

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, i, o, s=1):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(i,o,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,stride=s,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o*4,1,bias=False),nn.BatchNorm2d(o*4)); self.shortcut=(nn.Sequential(nn.Conv2d(i,o*4,1,stride=s,bias=False),nn.BatchNorm2d(o*4)) if s!=1 or i!=o*4 else nn.Identity()); self.relu=nn.ReLU(True)
    def forward(self, x): return self.relu(self.net(x)+self.shortcut(x))

_RESNET_CFG = {"resnet18": (BasicBlock,[2,2,2,2]), "resnet34": (BasicBlock,[3,4,6,3]), "resnet50": (Bottleneck,[3,4,6,3])}

def _make_layer(block, i, o, n, s):
    layers=[block(i,o,s)];
    for _ in range(1,n): layers.append(block(o*block.expansion,o))
    return nn.Sequential(*layers)

class ResNetDepth(nn.Module):
    def __init__(self, variant):
        super().__init__(); block,layers=_RESNET_CFG[variant]; exp=block.expansion; self.stem=nn.Sequential(nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(3,stride=2,padding=1)); self.layer1=_make_layer(block,64,64,layers[0],1); self.layer2=_make_layer(block,64*exp,128,layers[1],2); self.layer3=_make_layer(block,128*exp,256,layers[2],2); self.layer4=_make_layer(block,256*exp,512,layers[3],2); c=[64*exp,128*exp,256*exp,512*exp]; self.bottleneck=ConvBlock(c[3],c[3],0.3); self.up4=nn.ConvTranspose2d(c[3],c[3]//2,2,stride=2); self.dec4=ConvBlock(c[3]//2+c[2],c[3]//2); self.up3=nn.ConvTranspose2d(c[3]//2,c[2]//2,2,stride=2); self.dec3=ConvBlock(c[2]//2+c[1],c[2]//2); self.up2=nn.ConvTranspose2d(c[2]//2,c[1]//2,2,stride=2); self.dec2=ConvBlock(c[1]//2+c[0],c[1]//2); self.up1a=nn.ConvTranspose2d(c[1]//2,c[0]//2,2,stride=2); self.dec1=ConvBlock(c[0]//2,c[0]//2); self.up1b=nn.ConvTranspose2d(c[0]//2,c[0]//4,2,stride=2); self.dec0=ConvBlock(c[0]//4,c[0]//4); self.out=nn.Conv2d(c[0]//4,1,1)
    def forward(self, x):
        x=self.stem(x); s1=self.layer1(x); s2=self.layer2(s1); s3=self.layer3(s2); s4=self.layer4(s3); b=self.bottleneck(s4); d4=self.dec4(torch.cat([self.up4(b),s3],1)); d3=self.dec3(torch.cat([self.up3(d4),s2],1)); d2=self.dec2(torch.cat([self.up2(d3),s1],1)); d1=self.dec1(self.up1a(d2)); d0=self.dec0(self.up1b(d1)); return F.softplus(self.out(d0))

def build_simple_model(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in sd): sd = {k.replace("_orig_mod.","",1):v for k,v in sd.items()}
    if any("enc1" in k for k in sd): model = UNet()
    elif any("layer1" in k for k in sd):
        n_layers = max(int(k.split(".")[0].replace("layer","")) for k in sd if k.startswith("layer"))
        variant = {1:"resnet18",2:"resnet34_simple",3:"resnet50"}.get(n_layers,"resnet34_simple")
        bottleneck_check = any("net.6" in k for k in sd)
        model = ResNetDepth("resnet50" if bottleneck_check else ("resnet18" if sum(1 for k in sd if "layer1" in k and "net.0" in k) <= 4 else "resnet34"))
    else: raise ValueError(f"Cannot infer simple model arch from {ckpt_path}")
    model.load_state_dict(sd, strict=False); return model

@torch.no_grad()
def infer_simple(model, rgb_t):
    pred = model(rgb_t)
    if pred.dim() == 4: pred = pred.squeeze(1)
    return pred.squeeze(0).cpu().float().numpy()

# ─── ARCH: RESNET34 + EAGLE DECODER ──────────────────────────────────────────

import torchvision.models as tvm_models

class ResNet34EncoderWrapper(nn.Module):
    def __init__(self):
        super().__init__(); base=tvm_models.resnet34(weights=None); self.stem=nn.Sequential(base.conv1,base.bn1,base.relu,base.maxpool); self.layer1=base.layer1; self.layer2=base.layer2; self.layer3=base.layer3; self.layer4=base.layer4
    def forward(self, pixel_values, output_hidden_states=True):
        x=self.stem(pixel_values); h0=self.layer1(x); h1=self.layer2(h0); h2=self.layer3(h1); h3=self.layer4(h2)
        class _O:
            def __init__(s,v): s.hidden_states=v
        return _O((h0,h1,h2,h3))

class ASPPConv(nn.Sequential):
    def __init__(self, i, o, d): super().__init__(nn.Conv2d(i,o,3,padding=d,dilation=d,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))

class ASPPPooling(nn.Sequential):
    def __init__(self, i, o): super().__init__(nn.AdaptiveAvgPool2d(1),nn.Conv2d(i,o,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(self, x):
        sz=x.shape[-2:]
        for m in self: x=m(x)
        return F.interpolate(x,size=sz,mode="bilinear",align_corners=False)

class ASPP(nn.Module):
    def __init__(self, i, o, rates=(2,4,6)):
        super().__init__(); mods=[nn.Sequential(nn.Conv2d(i,o,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))]+[ASPPConv(i,o,r) for r in rates]+[ASPPPooling(i,o)]; self.convs=nn.ModuleList(mods); self.project=nn.Sequential(nn.Conv2d(len(self.convs)*o,o,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),nn.Dropout(0.1))
    def forward(self, x): return self.project(torch.cat([c(x) for c in self.convs],dim=1))

class DepthDecodeHeadEagle(nn.Module):
    def __init__(self, hidden_sizes, decoder_hidden_size=256, dropout=0.1, use_aspp=False):
        super().__init__(); self.num_stages=len(hidden_sizes); self.use_aspp=use_aspp; self.linear_c=nn.ModuleList()
        for i,c in enumerate(hidden_sizes): self.linear_c.append(nn.Identity() if (use_aspp and i==self.num_stages-1) else nn.Linear(c,decoder_hidden_size))
        if use_aspp: self.aspp=ASPP(hidden_sizes[-1],decoder_hidden_size)
        self.linear_fuse=nn.Conv2d(self.num_stages*decoder_hidden_size,decoder_hidden_size,kernel_size=1,bias=False); self.batch_norm=nn.BatchNorm2d(decoder_hidden_size); self.activation=nn.ReLU(True); self.dropout=nn.Dropout(p=dropout); self.depth_head=nn.Conv2d(decoder_hidden_size,1,kernel_size=1)
    def forward(self, enc_hidden, target_size):
        ref=enc_hidden[0].shape[-2:]; up=[]
        for i,(x,lin) in enumerate(zip(enc_hidden,self.linear_c)):
            if self.use_aspp and i==self.num_stages-1: xp=self.aspp(x)
            else:
                B,C,H,W=x.shape; xp=lin(x.permute(0,2,3,1).reshape(B,H*W,C)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
            if xp.shape[-2:]!=ref: xp=F.interpolate(xp,size=ref,mode="bilinear",align_corners=False)
            up.append(xp)
        x=self.linear_fuse(torch.cat(up,dim=1)); x=self.activation(self.batch_norm(x)); x=self.dropout(x); x=self.depth_head(x); x=F.interpolate(x,size=target_size,mode="bilinear",align_corners=False); return F.softplus(x).squeeze(1)

def build_resnet34_eagle(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False); sa=ckpt.get("args",{})
    enc=ResNet34EncoderWrapper(); enc.load_state_dict(ckpt["encoder_state_dict"])
    head=DepthDecodeHeadEagle([64,128,256,512],sa.get("decoder_hidden_size",256),sa.get("decoder_dropout",0.1),sa.get("use_aspp",False)); head.load_state_dict(ckpt["depth_head_state_dict"])
    return enc, head

@torch.no_grad()
def infer_encoder_head(enc, head, rgb_t):
    out=enc(pixel_values=rgb_t,output_hidden_states=True); H,W=rgb_t.shape[-2:]; pred=head(out.hidden_states,target_size=(H,W)); return pred.squeeze(0).cpu().float().numpy()

# ─── ARCH: SEGFORMER ─────────────────────────────────────────────────────────

from transformers import SegformerModel

class DepthDecodeHeadSeg(nn.Module):
    def __init__(self, hidden_sizes, decoder_hidden_size=256, dropout=0.1):
        super().__init__(); self.num_stages=len(hidden_sizes); self.linear_c=nn.ModuleList([nn.Linear(c,decoder_hidden_size) for c in hidden_sizes]); self.linear_fuse=nn.Conv2d(self.num_stages*decoder_hidden_size,decoder_hidden_size,kernel_size=1,bias=False); self.batch_norm=nn.BatchNorm2d(decoder_hidden_size); self.activation=nn.ReLU(True); self.dropout=nn.Dropout(p=dropout); self.depth_head=nn.Conv2d(decoder_hidden_size,1,kernel_size=1)
    def forward(self, enc_hidden, target_size):
        ref=enc_hidden[0].shape[-2:]; up=[]
        for x,lin in zip(enc_hidden,self.linear_c):
            B,C,H,W=x.shape; xp=lin(x.permute(0,2,3,1).reshape(B,H*W,C)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
            if xp.shape[-2:]!=ref: xp=F.interpolate(xp,size=ref,mode="bilinear",align_corners=False)
            up.append(xp)
        x=self.linear_fuse(torch.cat(up,dim=1)); x=self.activation(self.batch_norm(x)); x=self.dropout(x); x=self.depth_head(x); x=F.interpolate(x,size=target_size,mode="bilinear",align_corners=False); return F.softplus(x).squeeze(1)

def build_segformer(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False); sa=ckpt.get("args",{})
    seg_id=sa.get("seg_model_id","nvidia/segformer-b0-finetuned-ade-512-512")
    try: enc=SegformerModel.from_pretrained(seg_id)
    except Exception:
        from transformers import SegformerForSemanticSegmentation
        enc=SegformerForSemanticSegmentation.from_pretrained(seg_id).segformer
    enc.load_state_dict(ckpt["encoder_state_dict"],strict=False)
    head=DepthDecodeHeadSeg([32,64,160,256],sa.get("decoder_hidden_size",256),sa.get("decoder_dropout",0.1)); head.load_state_dict(ckpt["depth_head_state_dict"])
    return enc, head

@torch.no_grad()
def infer_segformer(enc, head, rgb_t):
    out=enc(pixel_values=rgb_t,output_hidden_states=True); H,W=rgb_t.shape[-2:]; pred=head(out.hidden_states,target_size=(H,W)); return pred.squeeze(0).cpu().float().numpy()

# ─── ARCH: MOBILEVIT ─────────────────────────────────────────────────────────

from transformers import MobileViTModel, MobileViTConfig

class DepthDecodeHeadMViT(nn.Module):
    def __init__(self, hidden_sizes, decoder_hidden_size=256, dropout=0.1):
        super().__init__(); self.num_stages=len(hidden_sizes); self.linear_c=nn.ModuleList([nn.Linear(c,decoder_hidden_size) for c in hidden_sizes]); self.linear_fuse=nn.Conv2d(self.num_stages*decoder_hidden_size,decoder_hidden_size,kernel_size=1,bias=False); self.batch_norm=nn.BatchNorm2d(decoder_hidden_size); self.activation=nn.ReLU(True); self.dropout=nn.Dropout(p=dropout); self.depth_head=nn.Conv2d(decoder_hidden_size,1,kernel_size=1)
    def forward(self, enc_hidden, target_size):
        ref=enc_hidden[0].shape[-2:]; up=[]
        for x,lin in zip(enc_hidden,self.linear_c):
            B,C,H,W=x.shape; xp=lin(x.permute(0,2,3,1).reshape(B,H*W,C)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
            if xp.shape[-2:]!=ref: xp=F.interpolate(xp,size=ref,mode="bilinear",align_corners=False)
            up.append(xp)
        x=self.linear_fuse(torch.cat(up,dim=1)); x=self.activation(self.batch_norm(x)); x=self.dropout(x); x=self.depth_head(x); x=F.interpolate(x,size=target_size,mode="bilinear",align_corners=False); return torch.sigmoid(x).squeeze(1)

def build_mobilevit(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False); sa=ckpt.get("args",{})
    hsd=ckpt["depth_head_state_dict"]; hidden_sizes=[]; i=0
    while f"linear_c.{i}.weight" in hsd: hidden_sizes.append(int(hsd[f"linear_c.{i}.weight"].shape[1])); i+=1
    if not hidden_sizes: hidden_sizes=[32,64,96,128,160]
    mvit_id=sa.get("mobilevit_model_id","apple/mobilevit-xx-small"); ckpt_dir=Path(ckpt_path).parent
    best_enc=ckpt_dir/"best_encoder"; cfg=MobileViTConfig.from_pretrained(str(best_enc) if best_enc.is_dir() else mvit_id)
    enc=MobileViTModel(cfg); enc.load_state_dict(ckpt["encoder_state_dict"],strict=False)
    head=DepthDecodeHeadMViT(hidden_sizes,sa.get("decoder_hidden_size",256),sa.get("decoder_dropout",0.1)); head.load_state_dict(ckpt["depth_head_state_dict"])
    return enc, head

@torch.no_grad()
def infer_mobilevit(enc, head, rgb_t):
    out=enc(pixel_values=rgb_t,output_hidden_states=True); n=head.num_stages; stage_h=out.hidden_states[-n:]; H,W=rgb_t.shape[-2:]; pred=head(stage_h,target_size=(H,W)); return pred.squeeze(0).cpu().float().numpy()

# ─── ARCH: MOBILENET ─────────────────────────────────────────────────────────

class _ASPPLite(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.conv1=nn.Conv2d(i,o,1,bias=False); self.conv2=nn.Conv2d(i,o,3,padding=6,dilation=6,bias=False); self.conv3=nn.Conv2d(i,o,3,padding=12,dilation=12,bias=False); self.gap_conv=nn.Conv2d(i,o,1,bias=False); self.project=nn.Sequential(nn.Conv2d(o*4,o,1,bias=False),nn.GroupNorm(8,o),nn.ReLU(True))
    def forward(self, x):
        h,w=x.shape[-2:]; gp=F.interpolate(self.gap_conv(F.adaptive_avg_pool2d(x,1)),size=(h,w),mode="bilinear",align_corners=False); return self.project(torch.cat([self.conv1(x),self.conv2(x),self.conv3(x),gp],dim=1))

class _UpBlock(nn.Module):
    def __init__(self, i, sk, o):
        super().__init__(); self.up=nn.ConvTranspose2d(i,i,4,stride=2,padding=1,bias=False); ia=i+sk; self.conv=nn.Sequential(nn.Conv2d(ia,o,3,padding=1,bias=False),nn.GroupNorm(min(8,o),o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1,bias=False),nn.GroupNorm(min(8,o),o),nn.ReLU(True))
    def forward(self, x, skip):
        x=self.up(x)
        if skip is not None:
            if x.shape[-2:]!=skip.shape[-2:]: x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)
            x=torch.cat([x,skip],dim=1)
        return self.conv(x)

class TinyDepthUNet(nn.Module):
    def __init__(self, bottleneck_channels=64, decoder_channels=(64,48,32,16), use_aspp=False):
        super().__init__(); mbn=tvm_models.mobilenet_v3_small(weights=None); old=mbn.features[0][0]; new=nn.Conv2d(5,old.out_channels,old.kernel_size,old.stride,old.padding,bias=(old.bias is not None))
        with torch.no_grad(): new.weight[:,:3]=old.weight; new.weight[:,3:]=0.0
        mbn.features[0][0]=new; self.enc_stem=mbn.features[0]; self.enc_s4=mbn.features[1]; self.enc_s8=nn.Sequential(*list(mbn.features[2:4])); self.enc_s16=nn.Sequential(*list(mbn.features[4:9])); self.enc_s32=nn.Sequential(*list(mbn.features[9:12]))
        self.use_aspp=bool(use_aspp); self.aspp=(_ASPPLite(96,bottleneck_channels) if use_aspp else nn.Sequential(nn.Conv2d(96,bottleneck_channels,1,bias=False),nn.GroupNorm(min(8,bottleneck_channels),bottleneck_channels),nn.ReLU(True)))
        self.eams=nn.ModuleDict(); c1,c2,c3,c4=decoder_channels; self.up1=_UpBlock(bottleneck_channels,48,c1); self.up2=_UpBlock(c1,24,c2); self.up3=_UpBlock(c2,16,c3); self.up4=_UpBlock(c3,0,c4); self.head=nn.Conv2d(c4,1,1)
        self.register_buffer("imagenet_mean",_IMAGENET_MEAN_T.clone(),persistent=False); self.register_buffer("imagenet_std",_IMAGENET_STD_T.clone(),persistent=False)
    def _coords(self, rgb):
        b,_,h,w=rgb.shape; y=torch.linspace(-1,1,h,device=rgb.device,dtype=rgb.dtype).view(1,1,h,1).expand(b,1,h,w); x=torch.linspace(-1,1,w,device=rgb.device,dtype=rgb.dtype).view(1,1,1,w).expand(b,1,h,w); return torch.cat([x,y],dim=1)
    def forward(self, rgb):
        b,_,h,w=rgb.shape; xi=torch.cat([rgb,self._coords(rgb)],dim=1); x2=self.enc_stem(xi); s4=self.enc_s4(x2); s8=self.enc_s8(s4); s16=self.enc_s16(s8); s32=self.enc_s32(s16); bn=self.aspp(s32); d1=self.up1(bn,s16); d2=self.up2(d1,s8); d3=self.up3(d2,s4); d4=self.up4(d3,None); ld=F.interpolate(self.head(d4),size=(h,w),mode="bilinear",align_corners=False); return {"log_depth":ld}

def build_mobilenet(ckpt_path):
    ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd={k.replace("_orig_mod.","",1) if k.startswith("_orig_mod.") else k:v for k,v in ckpt["model_state_dict"].items()}
    model=TinyDepthUNet(64,(64,48,32,16),False); model.load_state_dict(sd,strict=False); return model

@torch.no_grad()
def infer_mobilenet(model, rgb_t):
    out=model(rgb_t); return torch.exp(out["log_depth"]).squeeze().cpu().float().numpy()

# ─── EVALUATION LOOP ─────────────────────────────────────────────────────────

def run_eval(name, ckpt_path, arch_type, input_norm):
    log.info("=" * 60)
    log.info("Evaluating: %s  [%s]", name, ckpt_path)

    if arch_type == "simple":
        model = build_simple_model(ckpt_path).to(DEVICE).eval()
    elif arch_type == "resnet34_eagle":
        enc, head = build_resnet34_eagle(ckpt_path); enc=enc.to(DEVICE).eval(); head=head.to(DEVICE).eval()
    elif arch_type == "segformer":
        enc, head = build_segformer(ckpt_path); enc=enc.to(DEVICE).eval(); head=head.to(DEVICE).eval()
    elif arch_type == "mobilevit":
        enc, head = build_mobilevit(ckpt_path); enc=enc.to(DEVICE).eval(); head=head.to(DEVICE).eval()
    elif arch_type == "mobilenet":
        model = build_mobilenet(ckpt_path).to(DEVICE).eval()

    mat_file, images_ds, depths_ds, N_total = open_mat(MAT_PATH)
    indices = load_eigen_indices(SPLITS_PATH, N_total) if USE_SPLIT else list(range(N_total))

    all_metrics: List[Dict] = []; skipped = 0
    for step, idx in enumerate(indices):
        rgb, gt = read_sample(images_ds, depths_ds, idx)
        rgb_t = preprocess(rgb, INPUT_SIZE, input_norm).to(DEVICE)
        if arch_type == "simple":          pred = infer_simple(model, rgb_t)
        elif arch_type == "resnet34_eagle":pred = infer_encoder_head(enc, head, rgb_t)
        elif arch_type == "segformer":     pred = infer_segformer(enc, head, rgb_t)
        elif arch_type == "mobilevit":     pred = infer_mobilevit(enc, head, rgb_t)
        elif arch_type == "mobilenet":     pred = infer_mobilenet(model, rgb_t)
        gt_h, gt_w = gt.shape
        if pred.shape != gt.shape: pred = cv2.resize(pred, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
        if USE_CROP: pred, gt = eigen_crop(pred), eigen_crop(gt)
        m = compute_metrics(pred, gt)
        if m is None: skipped += 1; continue
        all_metrics.append(m)

    mat_file.close()
    if not all_metrics: log.error("No valid samples for %s", name); return
    silog   = float(np.mean([m["silog"]   for m in all_metrics]))
    abs_rel = float(np.mean([m["abs_rel"] for m in all_metrics]))
    delta1  = float(np.mean([m["delta1"]  for m in all_metrics]))
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  SILog  (↓) : {silog:.4f}")
    print(f"  AbsRel (↓) : {abs_rel:.4f}")
    print(f"  δ₁     (↑) : {delta1:.4f}")
    print(f"  skipped    : {skipped}/{len(indices)}")
    print(f"{'='*60}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NYUv2 benchmark inference + evaluation")
    p.add_argument("--data_dir", default=_NYU_DIR,
                   help="Dir holding nyu_depth_v2_labeled.mat + splits.mat.")
    p.add_argument("--models_dir", default=MODELS_DIR,
                   help="Root the registry checkpoint paths are resolved against.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MAT_PATH = os.path.join(args.data_dir, "nyu_depth_v2_labeled.mat")
    SPLITS_PATH = os.path.join(args.data_dir, "splits.mat")
    for name, ckpt, arch, norm in MODELS:
        ckpt_path = os.path.join(args.models_dir, ckpt)
        if not Path(ckpt_path).exists():
            log.warning("Skipping %s — checkpoint not found: %s", name, ckpt_path); continue
        try: run_eval(name, ckpt_path, arch, norm)
        except Exception as e: log.error("Failed %s: %s", name, e)
    print("\nAll evaluations complete.")
