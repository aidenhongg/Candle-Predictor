import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from pipeline.model_t import TransformerBCE
import hyperparams as hp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_lr_multiplier(current_step: int):
    """Compute learning rate multiplier for linear warmup + cosine annealing with warm restarts."""
    if current_step < hp.WARMUP:
        return current_step / max(1, hp.WARMUP)

    s = current_step - hp.WARMUP
    cycle_len = hp.T0
    cycle_start = 0

    while s >= cycle_start + cycle_len:
        cycle_start += cycle_len
        cycle_len *= hp.T_MULT

    progress = (s - cycle_start) / max(1, cycle_len)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_seed(seed=None):
    """Set random seeds for reproducibility. Returns the seed used."""
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def train(train_data: Dataset, test_data: Dataset, task: str = 'classifier', debug_mode: bool = False):
    """Train a transformer model with early stopping. Returns loss history and best predictions."""
    seed = set_seed(hp.SEED)
    savename = f"LR{hp.LEARNING_RATE}_DECAY{hp.WEIGHT_DECAY}_WARMUP{hp.WARMUP}_SEED{hp.SEED}"
    loss_record = []

    training_loader = DataLoader(
        train_data,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    testing_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    if task == 'classifier':
        model = TransformerBCE(
            embed_dim=320, num_heads=8, num_layers=8, ff_dim=768,
            head_size=1, debug=debug_mode,
        ).to(DEVICE)
    elif task == 'regressor':
        model = TransformerBCE(
            embed_dim=320, num_heads=8, num_layers=8, ff_dim=768,
            head_size=3, debug=debug_mode,
        ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=hp.LEARNING_RATE,
        betas=(hp.BETA1, hp.BETA2),
        weight_decay=hp.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss() if task == "classifier" else nn.MSELoss()
    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    # Initial evaluation before training
    stats, predictions = evaluate(model, testing_loader, criterion, task)
    _print_eval_stats(task, 0, stats)

    epochs, bad_epochs, lowest_loss = 0, 0, float('inf')
    best_predictions = predictions

    try:
        while bad_epochs < hp.PATIENCE:
            epochs += 1
            model.train()
            total_loss = 0

            for batch_count, (window_batch, label_batch) in enumerate(training_loader):
                optimizer.zero_grad()
                logits = model(window_batch)
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * len(window_batch)
                if batch_count % 500 == 0:
                    loss_record.append(loss.detach())
                    print(f"  Batch {batch_count}: loss {loss.item():.8f}")

            avg_train_loss = total_loss / len(training_loader.dataset)
            print(f"Epoch {epochs}: training loss {avg_train_loss:.8f}")

            stats, predictions = evaluate(model, testing_loader, criterion, task)

            if stats['avg_loss'] < lowest_loss - hp.STOP_BUFFER:
                lowest_loss = stats['avg_loss']
                bad_epochs = 0

                torch.save(model.state_dict(), f"./pipeline/models/{task}/{savename}.pt")

                with open(f"./pipeline/models/{task}/{savename}_stats.json", "w") as f:
                    recipe = {
                        "window size": hp.WINDOW_SIZE, "batch size": hp.BATCH_SIZE,
                        "learning rate": hp.LEARNING_RATE, "beta1": hp.BETA1, "beta2": hp.BETA2,
                        "weight decay": hp.WEIGHT_DECAY, "warmup": hp.WARMUP,
                        "T0": hp.T0, "T_mult": hp.T_MULT, "dropout": hp.DROPOUT,
                        "stop buffer": hp.STOP_BUFFER, "patience": hp.PATIENCE,
                        "vel alpha": hp.VEL_ALPHA, "accel alpha": hp.ACCEL_ALPHA,
                        "seed": seed,
                    }
                    json.dump({"recipe": recipe, "performance": stats}, f, indent=4)

                best_predictions = predictions
            else:
                bad_epochs += 1

            _print_eval_stats(task, epochs, stats)

    except KeyboardInterrupt:
        print("Training interrupted!")
        best_predictions = predictions

    return loss_record, best_predictions


def _print_eval_stats(task: str, epoch: int, stats: dict):
    """Print formatted evaluation statistics for the current epoch."""
    if task == 'classifier':
        print(f"Epoch {epoch}: val loss {stats['avg_loss']:.8f}, accuracy {stats['accuracy']:.2f}%")
        print(f"  False positives: {stats['false_positives']:.0f}")
        print(f"  False negatives: {stats['false_negatives']:.0f}")
    elif task == 'regressor':
        print(f"Epoch {epoch}: val loss {stats['avg_loss']:.8f}")
        print(f"  Avg high delta: {stats['average_high_d']:.8f}")
        print(f"  Avg low delta: {stats['average_low_d']:.8f}")
        print(f"  Avg close delta: {stats['average_close_d']:.8f}")


def evaluate(model, dataloader, criterion, task='classifier'):
    """Evaluate the model on a dataset. Returns stats dict and all predictions."""
    model.eval()
    all_predictions = []
    total_loss = 0
    correct_predictions = 0
    average_delta = torch.zeros(3, device=DEVICE)

    with torch.inference_mode():
        for window_batch, label_batch in dataloader:
            logits = model(window_batch)
            loss = criterion(logits, label_batch)
            total_loss += loss.item() * len(window_batch)

            if task == 'classifier':
                predictions = (torch.sigmoid(logits) >= 0.5).long()
                label_batch = label_batch.long()
                correct_predictions += (predictions == label_batch).sum().item()

                pred = predictions.view(-1)
                lab = label_batch.view(-1)

                TP = (pred * lab).sum().item()
                FP = (pred * (1 - lab)).sum().item()
                FN = ((1 - pred) * lab).sum().item()
                TN = ((1 - pred) * (1 - lab)).sum().item()

            elif task == 'regressor':
                predictions = torch.sinh(logits)
                label_batch = torch.sinh(label_batch)
                raw_delta = torch.abs(label_batch - predictions)
                average_delta += raw_delta.sum(dim=0).sum(dim=0)

            all_predictions.append(predictions)

    all_predictions = torch.vstack(all_predictions)
    avg_loss = total_loss / len(dataloader.dataset)

    if task == 'classifier':
        accuracy = 100 * correct_predictions / len(dataloader.dataset)
        false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0.0

        stats = {
            'avg_loss': avg_loss, 'accuracy': accuracy,
            'false_positive_rate': false_positive_rate, 'false_negative_rate': false_negative_rate,
            'false_positives': FP, 'false_negatives': FN,
        }
    else:
        average_delta = average_delta.cpu().numpy() / len(dataloader.dataset)
        stats = {
            'avg_loss': avg_loss,
            'average_high_d': float(average_delta[0]),
            'average_low_d': float(average_delta[1]),
            'average_close_d': float(average_delta[2]),
        }

    return stats, all_predictions
