import torch
from src.pipe.eval_pipe import eval_epoch


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    callbacks=None,
    global_step=0,
    gradient_accumulation_steps=1,
    eval_loader=None,
    eval_steps=None,
):
    model.train()
    total_loss, total_mlm, n_batches = 0, 0, 0

    optimizer.zero_grad()

    for step_idx, (input_ids, segment_ids, attn_mask, mlm_labels) in enumerate(loader):
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        attn_mask = attn_mask.to(device)
        mlm_labels = mlm_labels.to(device)

        # BF16 autocast mixed precision: unlike FP16, BF16 has FP32's exponent
        # range, so it doesn't need GradScaler / loss scaling to stay stable.
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = model(input_ids, segment_ids, mask=attn_mask, mlm_labels=mlm_labels)
            loss = out["loss"]
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

        loss.backward()

        # Optimizer step
        if (step_idx + 1) % gradient_accumulation_steps == 0 or (step_idx + 1) == len(loader):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            optimizer.zero_grad()

            lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            global_step += 1

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

            logged_loss = loss.item() * gradient_accumulation_steps

            if callbacks is not None:
                for cb in callbacks:
                    if hasattr(cb, "on_step_end"):
                        cb.on_step_end(
                            trainer=None,
                            step=global_step,
                            loss=logged_loss,
                            lr=lr,
                            grad_norm=grad_norm.item(),
                        )
                    if hasattr(cb, "log_metrics"):
                        cb.log_metrics(
                            {"mlm_acc": mlm_acc}, step=global_step, prefix="train"
                        )

            total_loss += logged_loss
            total_mlm += mlm_acc
            n_batches += 1

            # Mid-epoch evaluation
            if eval_loader is not None and eval_steps is not None and global_step % eval_steps == 0:
                eval_metrics = eval_epoch(model, eval_loader, device)
                print(f"\nStep {global_step} | val_loss={eval_metrics['loss']:.4f}  val_mlm_acc={eval_metrics['mlm_acc']:.3f}")
                if callbacks is not None:
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

    return {
        "loss": total_loss / max(n_batches, 1),
        "mlm_acc": total_mlm / max(n_batches, 1),
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
    gradient_accumulation_steps=1,
    eval_steps=None,
):
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
            callbacks=callbacks,
            global_step=global_step,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_loader=eval_loader,
            eval_steps=eval_steps,
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
