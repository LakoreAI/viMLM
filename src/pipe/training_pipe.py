import torch
from src.pipe.eval_pipe import eval_epoch


def train_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss, total_mlm, n_batches = 0, 0, 0

    for input_ids, segment_ids, attn_mask, mlm_labels in loader:
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        attn_mask = attn_mask.to(device)
        mlm_labels = mlm_labels.to(device)

        optimizer.zero_grad()

        # Autocast mixed precision
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(input_ids, segment_ids, mask=attn_mask, mlm_labels=mlm_labels)
            loss = out["loss"]

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # MLM accuracy on masked positions only
        masked = mlm_labels != -100
        if masked.any():
            mlm_acc = (
                (out["mlm_logits"].argmax(-1)[masked] == mlm_labels[masked])
                .float()
                .mean()
                .item()
            )
        else:
            mlm_acc = 0.0

        total_loss += loss.item()
        total_mlm += mlm_acc
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mlm_acc": total_mlm / n_batches,
    }


def train(
    model,
    train_loader,
    optimizer,
    device,
    epochs=5,
    scheduler=None,
    eval_loader=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler=scaler
        )

        log = (
            f"Epoch {epoch + 1}/{epochs}"
            f"  loss={train_metrics['loss']:.4f}"
            f"  mlm_acc={train_metrics['mlm_acc']:.3f}"
        )

        if eval_loader is not None:
            eval_metrics = eval_epoch(model, eval_loader, device)
            log += (
                f"  | val_loss={eval_metrics['loss']:.4f}"
                f"  val_mlm_acc={eval_metrics['mlm_acc']:.3f}"
            )

        print(log)

