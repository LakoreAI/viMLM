@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_mlm, total_nsp, n_batches = 0, 0, 0, 0

    for input_ids, segment_ids, mlm_labels, nsp_labels in loader:
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        mlm_labels = mlm_labels.to(device)
        nsp_labels = nsp_labels.to(device)

        out = model(input_ids, segment_ids, mlm_labels, nsp_labels)
        loss = out["loss"]

        masked = mlm_labels != -100
        mlm_acc = (
            (out["mlm_logits"].argmax(-1)[masked] == mlm_labels[masked])
            .float()
            .mean()
            .item()
        )
        nsp_acc = (out["nsp_logits"].argmax(-1) == nsp_labels).float().mean().item()

        total_loss += loss.item()
        total_mlm += mlm_acc
        total_nsp += nsp_acc
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mlm_acc": total_mlm / n_batches,
        "nsp_acc": total_nsp / n_batches,
    }


def eval(model, loader, device):
    metrics = eval_epoch(model, loader, device)
    print(
        f"Eval  loss={metrics['loss']:.4f}"
        f"  mlm_acc={metrics['mlm_acc']:.3f}"
        f"  nsp_acc={metrics['nsp_acc']:.3f}"
    )
    return metrics
