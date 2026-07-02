import os
import torch

class CheckpointCallback:
    def __init__(
        self,
        model,
        checkpoint_dir: str,
        save_best: bool = True,
        save_last: bool = True,
        save_steps: int = None,
        wandb_callback = None,
    ):
        """
        Callback for model checkpointing.
        """
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.save_best = save_best
        self.save_last = save_last
        self.save_steps = save_steps
        self.wandb_callback = wandb_callback

        self.best_loss = float("inf")

        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_step_end(
        self, trainer, step: int, loss: float, lr: float, grad_norm: float = None
    ) -> None:
        if self.save_steps and step > 0 and step % self.save_steps == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-{step}.pt")
            self._save_checkpoint(checkpoint_path)
            print(f"Saved step checkpoint to {checkpoint_path}")
            if self.wandb_callback is not None:
                try:
                    self.wandb_callback.log_artifact(checkpoint_path, name=f"checkpoint-{step}")
                except Exception as e:
                    print(f"Warning: Failed to log checkpoint to W&B: {e}")

    def on_evaluate(self, trainer, step: int, metrics: dict) -> None:
        loss = metrics.get("loss")
        if self.save_best and loss is not None:
            if loss < self.best_loss:
                self.best_loss = loss
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self._save_checkpoint(best_path)
                print(f"Saved new best model checkpoint to {best_path} with loss: {loss:.4f}")
                if self.wandb_callback is not None:
                    try:
                        self.wandb_callback.log_artifact(best_path, name="best-model")
                    except Exception as e:
                        print(f"Warning: Failed to log best model to W&B: {e}")

    def on_train_end(self, trainer) -> None:
        if self.save_last:
            last_path = os.path.join(self.checkpoint_dir, "last_model.pt")
            self._save_checkpoint(last_path)
            print(f"Saved last model checkpoint to {last_path}")
            if self.wandb_callback is not None:
                try:
                    self.wandb_callback.log_artifact(last_path, name="last-model")
                except Exception as e:
                    print(f"Warning: Failed to log last model to W&B: {e}")

    def _save_checkpoint(self, path: str):
        # Unwrap torch.compile wrapper if active
        model_to_save = self.model
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod

        torch.save({
            "model_state_dict": model_to_save.state_dict(),
            "config": getattr(model_to_save, "cfg", None),
        }, path)
