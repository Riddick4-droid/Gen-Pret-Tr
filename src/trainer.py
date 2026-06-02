import os
import math
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.logger import get_logger
from src.exceptions import ProjectException
from src.losses.loss import compute_loss
from src.compute_metrics.metrics import perplexity
from src.utils import get_resource_info

logger = get_logger(__name__)

class Trainer:
    """
    Handles training and validation loops for pre‑training and fine‑tuning.
    """
    def __init__(self, model, tokenizer, config, stage="pretrain"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.stage = stage                # "pretrain" or "finetune"
        self.cfg = config[stage]          # specific hyperparams
        self.device = torch.device(config["project"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) #put model on the correct device

        self.scaler = GradScaler('cuda') if self.cfg.get("mixed_precision", True) else None #initialize gradient scaler for mixed precision if enabled
        self.checkpoint_dir = config["paths"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, f"best_{stage}.pth")
        self.final_model_path = os.path.join(self.checkpoint_dir, f"{stage}_final.pth")

        self.use_mlflow = config.get("mlflow", {}).get("enabled", False)
        if self.use_mlflow:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
            self.mlflow.set_experiment(config["mlflow"]["experiment_name"])

        self._log_resource()

    def _log_resource(self):
        """Log compute resource info at the start of training."""
        info = get_resource_info()
        logger.info(f"Resource: {info}")
        if self.use_mlflow:
            try:
                self.mlflow.log_params({"resource": info.get("gpu_name", "CPU"), "gpu_count": info.get("gpu_count", 0)})
            except:
                pass

    def _configure_optimizer_and_scheduler(self, dataloader_len):
        """Create optimizer and OneCycleLR scheduler from config."""
        lr = self.cfg["learning_rate"]
        weight_decay = self.cfg["weight_decay"]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        grad_accum = self._grad_accum_steps()
        total_steps = dataloader_len * self.cfg["num_epochs"] // grad_accum
        warmup_steps = int(self.cfg["warmup_ratio"] * total_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0,
            anneal_strategy='cos'
        )
        return optimizer, scheduler

    def _grad_accum_steps(self):
        """Calculate how many steps to accumulate gradients based on effective and micro batch sizes.
        if small effective batch size is desired but GPU memory is limited, 
        we can accumulate gradients over multiple steps before updating model weights."""
        effective = self.cfg["effective_batch_size"]
        micro = self.cfg["micro_batch_size"]
        assert effective % micro == 0, "effective_batch_size must be divisible by micro_batch_size"
        return effective // micro

    def _train_one_epoch(self, dataloader, optimizer, scheduler, grad_accum):
        """Train for one epoch, returning average loss and perplexity.
        this is later called under n_epochs loop in the main train() method. 
        It handles the actual training logic for one epoch, including forward pass, loss computation, 
        backward pass with gradient scaling if enabled, optimizer step, and learning rate scheduling. 
        It also accumulates loss and token counts to compute average loss and perplexity at the end of the epoch."""
        self.model.train()
        total_loss_sum = 0.0
        total_tokens = 0
        pbar = tqdm(dataloader, desc=f"{self.stage.capitalize()} Train", leave=True)

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.amp.autocast('cuda'):
                logits, _, _ = self.model(input_ids, attention_mask) #the model returns logits, atten_weights_list, atten_weights_dict but we only need logits for loss computation here
                loss = compute_loss(logits, labels) / grad_accum #normalize loss by grad_accum to average it over the effective batch

            self.scaler.scale(loss).backward() #scale the loss for mixed precision to prevent underflow during backpropagation

            if (step + 1) % grad_accum == 0: #time to update weights after accumulating gradients for grad_accum steps
                self.scaler.unscale_(optimizer) #unscale gradients before clipping to get correct values
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"]) #clip gradients to prevent exploding gradients, which can destabilize training
                self.scaler.step(optimizer) #update model weights based on scaled gradients
                self.scaler.update() #update the scale factor for next iteration based on whether gradients overflowed
                scheduler.step() #update learning rate according to OneCycleLR schedule
                optimizer.zero_grad() #reset gradients for next accumulation cycle

            # loss accumulation weighted by tokens
            batch_loss = loss.item() * grad_accum # un-normalize the loss to get the total loss for this batch before averaging
            non_ignored = (labels != -100).sum().item() # count how many tokens in this batch contribute to the loss (labels != -100)
            total_loss_sum += batch_loss * non_ignored
            total_tokens += non_ignored

            avg_loss = total_loss_sum / total_tokens if total_tokens > 0 else 0.0
            ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
            pbar.set_postfix({"loss": f"{batch_loss:.3f}", "ppl": f"{ppl:.1f}"})

        epoch_loss = total_loss_sum / total_tokens
        epoch_ppl = perplexity(epoch_loss) #compute perplexity using the function from metrics.py 
        return epoch_loss, epoch_ppl

    @torch.no_grad()
    def _validate(self, dataloader):
        self.model.eval()
        total_loss_sum = 0.0
        total_tokens = 0
        pbar = tqdm(dataloader, desc=f"{self.stage.capitalize()} Val", leave=True)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.amp.autocast('cuda'):
                logits, _, _ = self.model(input_ids, attention_mask)
                loss = compute_loss(logits, labels)

            non_ignored = (labels != -100).sum().item()
            total_loss_sum += loss.item() * non_ignored
            total_tokens += non_ignored

        epoch_loss = total_loss_sum / total_tokens
        epoch_ppl = math.exp(epoch_loss) if epoch_loss < 100 else float('inf')
        return epoch_loss, epoch_ppl

    def train(self, train_loader, valid_loader):
        """Main training loop with early stopping and checkpointing."""
        optimizer, scheduler = self._configure_optimizer_and_scheduler(len(train_loader))
        grad_accum = self._grad_accum_steps()
        best_val_ppl = float('inf')
        patience = 0
        max_patience = self.cfg["early_stop_patience"]

        logger.info(f"Starting {self.stage}: {self.cfg['num_epochs']} epochs, "
                    f"effective batch {self.cfg['effective_batch_size']}, "
                    f"grad_accum {grad_accum}, device {self.device}")

        if self.use_mlflow:
            try:
                self.mlflow.start_run()
                self.mlflow.log_params({
                    "stage": self.stage,
                    "model_config": self.config["model"],
                    **self.cfg
                })
            except Exception as e:
                logger.warning(f"MLflow not reachable: {e}")

        for epoch in range(1, self.cfg["num_epochs"] + 1):
            logger.info(f"=== Epoch {epoch}/{self.cfg['num_epochs']} ===")
            train_loss, train_ppl = self._train_one_epoch(train_loader, optimizer, scheduler, grad_accum)
            logger.info(f"Train loss: {train_loss:.4f}, ppl: {train_ppl:.1f}")

            val_loss, val_ppl = self._validate(valid_loader)
            logger.info(f"Val loss: {val_loss:.4f}, ppl: {val_ppl:.1f}")

            if self.use_mlflow:
                try:
                    self.mlflow.log_metrics({
                        "train_loss": train_loss, "train_ppl": train_ppl,
                        "val_loss": val_loss, "val_ppl": val_ppl
                    }, step=epoch)
                except: 
                    pass

            # Early stopping & checkpoint
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                patience = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"✓ Best model saved (ppl={best_val_ppl:.1f})")
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"Early stopping after {epoch} epochs. Best ppl: {best_val_ppl:.1f}")
                    break

        torch.save(self.model.state_dict(), self.final_model_path)
        logger.info(f"Final model saved to {self.final_model_path}")

        if self.use_mlflow:
            try:
                self.mlflow.log_artifact(self.best_model_path)
                self.mlflow.end_run()
            except: 
                pass