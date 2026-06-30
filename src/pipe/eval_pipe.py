import torch


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_mlm, n_batches = 0, 0, 0

    for input_ids, segment_ids, attn_mask, mlm_labels in loader:
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        attn_mask = attn_mask.to(device)
        mlm_labels = mlm_labels.to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(input_ids, segment_ids, mask=attn_mask, mlm_labels=mlm_labels)
            loss = out["loss"]

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


def eval(model, loader, device, callbacks=None):
    metrics = eval_epoch(model, loader, device)
    print(f"Eval  loss={metrics['loss']:.4f}  mlm_acc={metrics['mlm_acc']:.3f}")
    if callbacks is not None:
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for cb in callbacks:
            if hasattr(cb, "on_evaluate"):
                cb.on_evaluate(
                    trainer=None,
                    step=0,
                    metrics={
                        "loss": metrics["loss"],
                        "accuracy": metrics["mlm_acc"],
                    },
                )
    return metrics
