"""Resumable training with AMP, correctly normalized accumulation, and EMA."""
import gc
import json
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.nn import functional as F

from .data import load_prepared
from .models import ModelEMA, build_model, build_optimizer, make_loader
from .runtime import selected_amp_dtype
from .utils import (atomic_torch_save, capture_rng, digest_json, restore_rng,
                    seed_everything, source_fingerprint, write_json)


def amp_context(device, enabled=True):
    """Choose CUDA autocast with the same native policy as GradScaler."""
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = selected_amp_dtype(device, enabled)
    return torch.autocast(device_type="cuda", dtype=dtype)


def metric_dict(labels, probabilities):
    """Compute accuracy, macro-F1 and multiclass log-loss."""
    predictions = probabilities.argmax(1)
    count = probabilities.shape[1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(count)), average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=list(range(count)))),
    }


@torch.inference_mode()
def predict_loader(model, loader, device, amp=True):
    """Collect finite probabilities, labels and IDs in loader order."""
    model.eval()
    probabilities, labels, ids = [], [], []
    for images, target, batch_ids in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.contiguous_format)
        with amp_context(device, amp):
            logits = model(images)
        probability = logits.float().softmax(1).cpu().numpy()
        if not np.isfinite(probability).all():
            raise FloatingPointError("Non-finite inference probabilities")
        probabilities.append(probability)
        labels.extend(target.tolist())
        ids.extend(batch_ids)
    return np.concatenate(probabilities), np.asarray(labels, dtype=np.int64), ids


def lr_factor(step, total_steps, warmup_steps, minimum):
    """Return the warmup/cosine multiplier for an optimizer step."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1., max(0., (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)))
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, ema, loader, optimizer, scaler, device, cfg, epoch, global_step, total_steps, warmup_steps):
    """Normalize accumulated gradients by actual samples, including short tails."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_correct = total_seen = window_samples = 0
    finetune = epoch >= cfg["finetune_epoch"]
    smoothing = cfg["finetune_label_smoothing"] if finetune else cfg["label_smoothing"]
    accumulation = int(cfg["accumulation_steps"])
    for batch_index, (images, labels, _) in enumerate(loader):
        if batch_index % accumulation == 0:
            factor = lr_factor(global_step, total_steps, warmup_steps, cfg["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
        images = images.to(device, non_blocking=True, memory_format=torch.contiguous_format)
        labels = labels.to(device, non_blocking=True)
        mixed = not finetune and cfg["mixup_alpha"] > 0 and np.random.random() < cfg["mixup_probability"]
        lam, other = 1., labels
        if mixed:
            lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
            permutation = torch.randperm(len(images), device=device)
            images = lam * images + (1-lam) * images[permutation]
            other = labels[permutation]
        with amp_context(device, cfg["amp"]):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum", label_smoothing=smoothing)
            if mixed:
                loss = lam * loss + (1-lam) * F.cross_entropy(logits, other, reduction="sum", label_smoothing=smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss tidak finite pada epoch {epoch}, batch {batch_index}")
        scaler.scale(loss).backward()
        count = len(labels)
        window_samples += count
        total_seen += count
        total_loss += float(loss.detach())
        prediction = logits.detach().argmax(1)
        total_correct += lam * prediction.eq(labels).sum().item() + (1-lam) * prediction.eq(other).sum().item()
        boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        if boundary:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                ema.update(model)
                global_step += 1
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0
        if (batch_index + 1) % 100 == 0:
            print(f"epoch {epoch} | batch {batch_index+1}/{len(loader)} | loss {total_loss/total_seen:.4f}", flush=True)
    return {"train_loss": total_loss / total_seen, "train_soft_accuracy": total_correct / total_seen}, global_step


def cpu_state(model):
    """Clone model parameters and buffers to CPU for serialization."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_model(project, cfg, run_name=None, max_hours=None, stop_after=None):
    """Train/resume one fixed experiment; max_hours pauses only after an epoch."""
    project = Path(project)
    cfg = json.loads(json.dumps(cfg))
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["accumulation_steps"] < 1:
        raise ValueError("epochs, batch_size, accumulation_steps must be positive")
    if not (0 < cfg["finetune_epoch"] <= cfg["epochs"]):
        raise ValueError("finetune_epoch must be within the planned epoch horizon")
    train, valid, test, sample, manifest = load_prepared(project)
    if len(manifest["class_names"]) != cfg["num_classes"]:
        raise ValueError("Number of classes differs from prepared data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["require_gpu"] and device.type != "cuda":
        raise RuntimeError("Aktifkan GPU di Kaggle Settings. CPU hanya untuk smoke test, bukan training penuh.")
    torch.set_num_threads(cfg["cpu_threads"])
    seed_everything(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = project / "outputs" / (run_name or f"{cfg['model']}_seed{cfg['seed']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = digest_json(cfg)
    source = source_fingerprint()
    last_path = run_dir / "last.pt"
    checkpoint = None
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["config_hash"] != config_hash or checkpoint["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Config/data berubah. Gunakan RUN_TAG baru; jangan mencampur eksperimen.")
        if checkpoint["source_sha256"] != source:
            raise ValueError("Source training berubah sejak checkpoint. Gunakan run baru.")
        if checkpoint["finished"]:
            print("Run sudah selesai; memakai checkpoint tersimpan:", run_dir, flush=True)
            return run_dir
    elif any(run_dir.iterdir()):
        raise FileExistsError("Folder run terisi tanpa last.pt. Pilih RUN_TAG baru atau periksa output yang ada.")
    model = build_model(cfg, initialize_pretrained=checkpoint is None).to(device, memory_format=torch.contiguous_format)
    ema = ModelEMA(model, cfg["ema_decay"])
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=selected_amp_dtype(device, cfg["amp"]) == torch.float16)
    steps_per_epoch = math.ceil(math.ceil(len(train) / cfg["batch_size"]) / cfg["accumulation_steps"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = min(total_steps - 1, steps_per_epoch * cfg["warmup_epochs"])
    start, global_step, best_accuracy, best_loss, stale = 1, 0, -1., float("inf"), 0
    history = []
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        ema.model.load_state_dict(checkpoint["ema_state"])
        ema.updates = checkpoint["ema_updates"]
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng(checkpoint["rng"])
        start, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best_accuracy, best_loss, stale = checkpoint["best_accuracy"], checkpoint["best_loss"], checkpoint["stale"]
        history = checkpoint["history"]
    del checkpoint
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
               "torchvision": str(torchvision.__version__), "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "note": "Seeds fixed; CUDA kernels need not be bitwise deterministic."})
    started = time.perf_counter()
    print(f"{cfg['model']} | train={len(train):,}, valid={len(valid):,} | {device} | start epoch={start}", flush=True)
    for epoch in range(start, cfg["epochs"] + 1):
        epoch_started = time.perf_counter()
        fine = epoch >= cfg["finetune_epoch"]
        size = cfg["finetune_size"] if fine else cfg["image_size"]
        loader = make_loader(train, cfg, size, training=True, epoch=epoch, finetune=fine)
        train_metrics, global_step = train_epoch(model, ema, loader, optimizer, scaler, device, cfg, epoch,
                                                global_step, total_steps, warmup_steps)
        del loader
        loader = make_loader(valid, cfg, cfg["val_size"])
        probabilities, labels, ids = predict_loader(ema.model, loader, device, cfg["amp"])
        metrics = metric_dict(labels, probabilities)
        improved = (metrics["accuracy"] > best_accuracy or
                    (metrics["accuracy"] == best_accuracy and metrics["log_loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, stale = metrics["accuracy"], metrics["log_loss"], 0
        else:
            stale += 1
        row = dict(epoch=epoch, image_size=size, **train_metrics,
                   **{f"val_{k}": v for k, v in metrics.items()},
                   lr=optimizer.param_groups[0]["lr"], seconds=time.perf_counter()-epoch_started)
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        common = {
            "format_version": 2, "config": cfg, "config_hash": config_hash,
            "fingerprint": manifest["fingerprint"], "class_names": manifest["class_names"],
            "source_sha256": source, "epoch": epoch,
            "weights_origin": "torchvision ImageNet-1K" if cfg["pretrained"] else "random initialization (test/ablation)",
        }
        if improved:
            atomic_torch_save(run_dir / "best.pt", dict(common, model_state=cpu_state(ema.model),
                              weight_type="EMA", validation_metrics=metrics))
            np.savez_compressed(run_dir / "best_validation.npz", ids=np.asarray(ids, dtype=str), labels=labels,
                                probabilities=probabilities)
        early_stop = epoch >= cfg["min_epochs"] and stale >= cfg["patience"]
        finished = epoch == cfg["epochs"] or early_stop
        last = dict(common, model_state=cpu_state(model), ema_state=cpu_state(ema.model),
                    ema_updates=ema.updates, optimizer_state=optimizer.state_dict(),
                    scaler_state=scaler.state_dict(), rng=capture_rng(), global_step=global_step,
                    best_accuracy=best_accuracy, best_loss=best_loss, stale=stale,
                    history=history, finished=finished)
        atomic_torch_save(last_path, last)
        write_json(run_dir / "summary.json", {
            "model": cfg["model"], "completed_epoch": epoch, "planned_epochs": cfg["epochs"],
            "best_validation_accuracy": best_accuracy, "best_validation_log_loss": best_loss,
            "finished": finished, "early_stopped": early_stop,
            "score_type": "model-selection holdout; not independent final test / Kaggle score",
            "public_score": None, "private_score": None,
        })
        print(json.dumps(row), flush=True)
        del last, loader
        if finished:
            break
        if (stop_after is not None and epoch >= stop_after) or (max_hours and time.perf_counter()-started >= max_hours*3600):
            print("Run dijeda pada batas epoch; ulang sel training untuk melanjutkan last.pt.", flush=True)
            break
    del model, ema, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir


def load_inference(checkpoint_path):
    """Restore checkpoint weights without downloading pretrained weights."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(payload["config"], initialize_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    model = model.to(device, memory_format=torch.contiguous_format).eval()
    return model, payload, device
