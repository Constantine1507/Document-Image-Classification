"""ImageNet-only backbones, conservative page augmentation, and EMA."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .data import open_document


def default_config(name):
    """A strong starting recipe, not hyperparameters proven optimal on this data."""
    if name not in {"convnext_tiny", "efficientnet_v2_s", "smoke_cnn"}:
        raise ValueError(name)
    return {
        "model": name, "num_classes": 16, "pretrained": True,
        "local_imagenet_weights": None, "seed": 42,
        "epochs": 20, "finetune_epoch": 15,
        "image_size": 320, "finetune_size": 384, "val_size": 384,
        "eval_sizes": [384, 448], "batch_size": 8, "eval_batch_size": 16,
        "accumulation_steps": 4, "num_workers": 4,
        "lr": 1e-4 if name == "convnext_tiny" else 2e-4,
        "head_lr_multiplier": 5.0, "weight_decay": 0.01,
        "warmup_epochs": 2, "min_lr_ratio": 0.02,
        "label_smoothing": 0.05, "finetune_label_smoothing": 0.02,
        "mixup_alpha": 0.1, "mixup_probability": 0.25,
        "grad_clip": 1.0, "ema_decay": 0.999,
        "patience": 6, "min_epochs": 18,
        "amp": True, "cpu_threads": 2, "require_gpu": True,
        "clean_train_eval_rows": 2048,
    }


class Letterbox:
    """Keep the entire page and its aspect ratio with white padding."""
    def __init__(self, size):
        """Record the target square resolution."""
        self.size = int(size)

    def __call__(self, image):
        """Resize the whole page with aspect-preserving white padding."""
        return ImageOps.pad(image, (self.size, self.size),
                            method=Image.Resampling.BICUBIC, color="white")


def make_transform(size, training=False, finetune=False):
    """Preserve page layout with letterboxing and mild training augmentation."""
    operations = [Letterbox(size)]
    if training:
        operations += [
            transforms.RandomAffine(
                degrees=1 if finetune else 2,
                translate=(0.01, 0.01), scale=(0.98, 1.02) if finetune else (0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR, fill=255),
            transforms.ColorJitter(brightness=.05 if finetune else .1,
                                   contrast=.05 if finetune else .1),
        ]
    operations += [transforms.ToTensor(),
                   transforms.Normalize((.485, .456, .406), (.229, .224, .225))]
    return transforms.Compose(operations)


class DocumentDataset(Dataset):
    """Read images lazily from files or ZIP members with explicit IDs."""
    def __init__(self, frame, transform):
        """Store image references, labels and preprocessing for lazy reads."""
        self.paths = frame.resolved_path.astype(str).tolist()
        self.members = frame.zip_member.fillna("").astype(str).tolist()
        self.ids = frame.id.astype(str).tolist()
        self.labels = frame.label.astype(int).tolist() if "label" in frame else [-1] * len(frame)
        self.transform = transform

    def __len__(self):
        """Return the number of metadata rows in this dataset."""
        return len(self.ids)

    def __getitem__(self, index):
        """Decode and transform one document, retaining its label and ID."""
        try:
            image, _ = open_document(self.paths[index], self.members[index])
            return self.transform(image), self.labels[index], self.ids[index]
        except Exception as exc:
            raise RuntimeError(f"Gagal membaca ID={self.ids[index]}, path={self.paths[index]}") from exc


def seed_worker(worker_id):
    """Seed worker-local Python and NumPy RNGs from PyTorch."""
    import random
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(frame, cfg, size, training=False, epoch=0, finetune=False):
    """Epoch-specific workers/order allow clean resume at epoch boundaries."""
    workers = int(cfg["num_workers"])
    return DataLoader(
        DocumentDataset(frame, make_transform(size, training, finetune)),
        batch_size=cfg["batch_size"] if training else cfg["eval_batch_size"],
        shuffle=training, num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=False,
        generator=torch.Generator().manual_seed(int(cfg["seed"]) + epoch),
        worker_init_fn=seed_worker,
        **({"prefetch_factor": 2} if workers else {}),
    )


def build_model(cfg, initialize_pretrained=True):
    """Construct a classifier with explicit pretrained or test initialization."""
    name, count = cfg["model"], cfg["num_classes"]
    if name == "smoke_cnn":
        return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, count))
    factory, weights = {
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    }[name]
    use_pretrained = bool(cfg["pretrained"] and initialize_pretrained)
    local = cfg.get("local_imagenet_weights")
    try:
        model = factory(weights=weights if use_pretrained and not local else None)
        if use_pretrained and local:
            state = torch.load(Path(local), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"Gagal memuat {name}. Aktifkan Internet Kaggle untuk bobot ImageNet resmi, "
            "atau isi local_imagenet_weights dengan state_dict torchvision yang sesuai. "
            "Tidak ada fallback diam-diam ke bobot acak."
        ) from exc
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, count)
    return model


class ModelEMA:
    """Average parameters and floating buffers; copy integer counters exactly."""
    def __init__(self, model, decay=.999):
        """Create a frozen evaluation copy and initialize its update counter."""
        self.model = deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update floating averages and copy integer counters after an optimizer step."""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for name, average in self.model.state_dict().items():
            value = current[name].detach()
            if average.is_floating_point():
                average.mul_(decay).add_(value, alpha=1 - decay)
            else:
                average.copy_(value)


def build_optimizer(model, cfg):
    """Use a faster classifier LR and no decay for biases/norm parameters."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = cfg["head_lr_multiplier"] if name.startswith("classifier.") else 1.0
        decay = 0.0 if parameter.ndim <= 1 or name.endswith("bias") else cfg["weight_decay"]
        key = (float(multiplier), float(decay))
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([
        {"params": params, "lr": cfg["lr"] * multiplier,
         "base_lr": cfg["lr"] * multiplier, "weight_decay": decay}
        for (multiplier, decay), params in groups.items()
    ])
