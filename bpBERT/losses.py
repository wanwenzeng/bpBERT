import torch
from torch import nn

def multinomial_nll(true_counts, logits):
    logits_perm = logits.permute(0, 2, 1) 
    true_counts_perm = true_counts.permute(0, 2, 1)

    dist = torch.distributions.multinomial.Multinomial(total_count=10000000, logits=logits_perm)

    # Normalize by batch size. One could also normalize by
    # sequence length here.
    batch_size = float(true_counts.shape[0])

    return -torch.sum(dist.log_prob(true_counts_perm)) / batch_size

def mse(true, pred):
    counts_loss_f = torch.nn.MSELoss()
    mse_loss = counts_loss_f(pred, true)
    return mse_loss

def counts_multinomial_nll(true_counts, preds, c_task_weight=10, output_dir="", is_train=True):
    probs = preds / torch.sum(preds, dim=-1, keepdims=True)

    logits = torch.log(probs / (1 - probs))

    # multinomial loss
    multinomial_loss = multinomial_nll(true_counts, logits)

    counts_loss_f = torch.nn.MSELoss()

    mse_loss = counts_loss_f(torch.log(1 + torch.sum(true_counts, dim=-1)),
                            torch.log(1 + torch.sum(preds, dim=-1)))

    return multinomial_loss.mean() + c_task_weight * mse_loss.mean()

def binary_crossentropy(true, pred):
    loss_func =  nn.BCELoss()
    return loss_func(pred, true)
