import torch


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_mlm, total_nsp, n_batches = 0, 0, 0, 0

    for input_ids, segment_ids, mlm_labels, nsp_labels in loader:
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        mlm_labels = mlm_labels.to(device)
        nsp_labels = nsp_labels.to(device)

        optimizer.zero_grad()
        out = model(input_ids, segment_ids, mlm_labels, nsp_labels)

        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # MLM accuracy on masked positions only
        masked = mlm_labels != -100
        mlm_acc = (
            (out["mlm_logits"].argmax(-1)[masked] == mlm_labels[masked])
            .float()
            .mean()
            .item()
        )

        # NSP accuracy
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


def train(
    model, train_loader, optimizer, device, epochs=5, scheduler=None, eval_loader=None
):
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)

        log = (
            f"Epoch {epoch + 1}/{epochs}"
            f"  loss={train_metrics['loss']:.4f}"
            f"  mlm_acc={train_metrics['mlm_acc']:.3f}"
            f"  nsp_acc={train_metrics['nsp_acc']:.3f}"
        )

        if eval_loader is not None:
            eval_metrics = eval_epoch(model, eval_loader, device)
            log += (
                f"  | val_loss={eval_metrics['loss']:.4f}"
                f"  val_mlm_acc={eval_metrics['mlm_acc']:.3f}"
                f"  val_nsp_acc={eval_metrics['nsp_acc']:.3f}"
            )

        print(log)
