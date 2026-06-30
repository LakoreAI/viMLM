import torch
from src.pipe.eval_pipe import eval_epoch


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    scaler=None,
    callbacks=None,
    global_step=0,
):
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

        lr = optimizer.param_groups[0]["lr"]
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

        if callbacks is not None:
            for cb in callbacks:
                if hasattr(cb, "on_step_end"):
                    cb.on_step_end(
                        trainer=None, step=global_step, loss=loss.item(), lr=lr
                    )
                if hasattr(cb, "log_metrics"):
                    cb.log_metrics(
                        {"mlm_acc": mlm_acc}, step=global_step, prefix="train"
                    )

        total_loss += loss.item()
        total_mlm += mlm_acc
        n_batches += 1
        global_step += 1

    return {
        "loss": total_loss / n_batches,
        "mlm_acc": total_mlm / n_batches,
    }, global_step


def train(
    model,
    train_loader,
    optimizer,
    device,
    epochs=5,
    scheduler=None,
    eval_loader=None,
    callbacks=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if callbacks is None:
        callbacks = []
    elif not isinstance(callbacks, list):
        callbacks = [callbacks]

    for cb in callbacks:
        if hasattr(cb, "on_train_begin"):
            cb.on_train_begin(trainer=None)

    global_step = 0
    for epoch in range(epochs):
        train_metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler=scaler,
            callbacks=callbacks,
            global_step=global_step,
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

            for cb in callbacks:
                if hasattr(cb, "on_evaluate"):
                    cb.on_evaluate(
                        trainer=None,
                        step=global_step,
                        metrics={
                            "loss": eval_metrics["loss"],
                            "accuracy": eval_metrics["mlm_acc"],
                        },
                    )

        print(log)

    for cb in callbacks:
        if hasattr(cb, "on_train_end"):
            cb.on_train_end(trainer=None)
