from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn as nn
import torch
import math

from pipeline.model_t import TransformerBCE

from hyperparams import *
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_lr_mult(current_step : int):
    if current_step < WARMUP:
        return current_step / max(1, WARMUP)

    s = current_step - WARMUP
    cycle_len = T0
    cycle_start = 0

    while s >= cycle_start + cycle_len:
        cycle_start += cycle_len
        cycle_len *= T_MULT

    progress = (s - cycle_start) / max(1, cycle_len)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def set_seed(seed = None):
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed

def train(train_data : Dataset, test_data : Dataset, task : str = 'classifier', debug_mode = False):
    
    seed = set_seed(SEED)
    loss_record = []

    training_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    testing_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        drop_last=True,
        num_workers=0
    )
    if task == 'classifier':
        model = TransformerBCE(embed_dim = 128, num_heads = 8, num_layers=3, ff_dim=256, head_size = 1, debug = debug_mode).to(DEVICE)
        # model = TransformerBCE(head_size = 1, debug = debug_mode).to(DEVICE)
    elif task == 'regressor':
        model = TransformerBCE(embed_dim = 128, num_heads = 8, num_layers=3, ff_dim=256, head_size = 3, debug = debug_mode).to(DEVICE)
        # model = TransformerBCE(head_size = 3, debug = debug_mode).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), 
                            lr = LEARNING_RATE, 
                            betas = (BETA1, BETA2), 
                            weight_decay = WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss() if task == "classifier" else nn.MSELoss()

    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_mult)

    stats, predictions = evaluate(model, testing_loader, criterion, task)
    
    if task == 'classifier':
        print(f"Epoch {0}: validation loss of {stats['avg_loss']:.8f}, accuracy of {stats['accuracy']:.2f}%")
    elif task == 'regressor':
        print(f"Epoch {0}: validation loss of {stats['avg_loss']:.8f}\n average high delta of {stats['average_high_d']:.8f}\n average low delta of {stats['average_low_d']:.8f}\n average close delta of {stats['average_close_d']:.8f}")


    epochs, bad_epochs, lowest_loss = 0, 0, float('inf')
    try:
        while bad_epochs < PATIENCE:
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
                if not batch_count % 500:
                    loss_record.append(loss.detach())
                    print(f"{batch_count} batches processed with loss {loss.item():.8f}")

            print(f"Epoch {epochs}: training loss of {total_loss / len(training_loader.dataset):.8f}")

            stats, predictions = evaluate(model, testing_loader, criterion, task)
            lowest_loss = min(stats['avg_loss'], lowest_loss)

            if lowest_loss >= stats['avg_loss'] - STOP_BUFFER:
                bad_epochs += 1

                torch.save(model.state_dict(), f"./pipeline/models/{task}/model_{epochs}.pt")
                with open(f"./pipeline/models/{task}/model_{epochs}_stats.json", "w") as f:
                    dicta = {"recipe" : {"window size" : WINDOW_SIZE, "batch size" : BATCH_SIZE, 
                                         "learning rate" : LEARNING_RATE, "beta1" : BETA1, "beta2" : BETA2, 
                                         "weight decay" : WEIGHT_DECAY, "warmup" : WARMUP, 
                                         "T0" : T0, "T_mult" : T_MULT,
                                         "dropout" : DROPOUT,
                                         "stop buffer" : STOP_BUFFER, "patience" : PATIENCE, 
                                         "vel alpha" : VEL_ALPHA, "accel alpha" : ACCEL_ALPHA, 
                                         "seed" : seed},
                            "performance" : stats}
                    json.dump(dicta, f, indent=4)
                best_predictions = predictions

            if task == 'classifier':
                print(f"Epoch {epochs}: validation loss of {stats['avg_loss']:.8f}, accuracy of {stats['accuracy']:.2f}%")
                print(f"{stats['false_positives']:.0f} false positives")
                print(f"{stats['false_negatives']:.0f} false negatives")
            elif task == 'regressor':
                print(f"Epoch {epochs}: validation loss of {stats['avg_loss']:.8f}")
                print(f"Average high delta of {stats['average_high_d']:.8f}")
                print(f"average low delta of {stats['average_low_d']:.8f}:")
                print(f"average close delta of {stats['average_close_d']:.8f}")

    except KeyboardInterrupt:
        print("Training interrupted!")
        best_predictions = predictions

    return loss_record, best_predictions

def evaluate(model, dataloader, criterion, task = 'classifier'):
    model.eval()
    all_predictions = []
    total_loss = 0
    # for classifier
    correct_predictions = 0
    # for regressor
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
                predictions, label_batch = torch.sinh(logits), torch.sinh(label_batch)
                raw_delta = torch.abs(label_batch - predictions)
                average_delta += raw_delta.sum(dim=0).sum(dim=0)

            all_predictions.append(predictions)


    all_predictions = torch.vstack(all_predictions)
    avg_loss = total_loss / len(dataloader.dataset)

    if task == 'classifier':
        accuracy = 100 * correct_predictions / len(dataloader.dataset)
        false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0.0

        stats = {'avg_loss' : avg_loss, 'accuracy' : accuracy, 
                 'false_positive_rate' : false_positive_rate, 'false_negative_rate' : false_negative_rate,
                 'false_positives' : FP, 'false_negatives' : FN}
    else:
        average_delta = average_delta.cpu().numpy()
        average_delta /= len(dataloader.dataset)
    
        stats = {'avg_loss' : avg_loss, 
                 'average_high_d' : float(average_delta[0]), 'average_low_d' : float(average_delta[1]), 'average_close_d' : float(average_delta[2])}
        
    return stats, all_predictions
