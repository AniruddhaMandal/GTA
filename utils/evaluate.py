from collections import Counter
import torch

def train_epoch(model, dataloader, loss_fn, metric_fn, optimizer, scheduler, device, task=None):
    total_loss, total_acc = 0,0
    model.train()
    for i, batch in enumerate(dataloader):
        batch.to(device)
        optimizer.zero_grad()
        if(task == "inductive_edge"):
            y = batch.edge_label.to(torch.float32)
        else:
            y = batch.y
        y_pred = model(batch)
        loss = loss_fn(y_pred,y)
        loss.backward()
        optimizer.step()
        total_loss += loss

        if(task != "inductive_edge"):
            acc = metric_fn(y_pred,y)
            total_acc += acc
        else:
            acc = 0
    if scheduler != None:
        scheduler.step(total_loss)
    avg_loss = total_loss/(i+1)
    avg_acc = total_acc/(i+1)
    if isinstance(avg_loss,torch.Tensor):
        avg_loss = avg_loss.item()
    if isinstance(avg_acc, torch.Tensor):
        avg_acc = avg_acc.item()
    return avg_loss, avg_acc

@torch.no_grad()
def test_epoch(model, dataloader, loss_fn, metric_fn, device, task=None):
    avg_loss, avg_acc = 0, 0
    if task == "inductive_edge":
        avg_acc = None
    model.eval()
    for i, batch in enumerate(dataloader):
        batch.to(device)
        if(task == "inductive_edge"):
            y = batch.edge_label.to(torch.float32)
        else: 
            y = batch.y
        y_pred = model(batch)
        loss = loss_fn(y_pred, y)
        if(task == "inductive_edge"):
            acc = metric_fn(batch)
            avg_acc = dict(Counter(acc)+Counter(avg_acc)) 
        else:
            acc = metric_fn(y_pred, y)
            avg_acc += acc
        avg_loss += loss

    avg_loss = avg_loss/(i+1)
    avg_loss = avg_loss.item()
    if(task!="inductive_edge"):
        avg_acc = avg_acc/(i+1)
        avg_acc = avg_acc.item()
    else:
        avg_acc = {key: value/(i+1) for key, value in avg_acc.items()}
    return avg_loss, avg_acc