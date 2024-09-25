"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Model inference.
"""

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, train_dataloader, test_dataloader, optimizer, scheduler, criterion,
          num_epochs=100, starting_epoch=0, device='cuda'):
    """
    Trains the model on the train_dataloader for epochs. Evaluates the model
    after each epoch. Returns a checkpoint containing the model's state dict
    and current epoch.
    """
    model = model.to(device)
    criterion = criterion.to(device)

    epoch = starting_epoch
    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        model.train()            
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.
        progress_bar = tqdm(train_dataloader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
    
            images = images.to(device)
            labels = labels.to(device)
    
            model.zero_grad()
            pred = model(images)
    
            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            optimizer.step()
            xentropy_loss_avg += xentropy_loss.item()
    
            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total
    
            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        test_acc, outputs = test(model, test_dataloader)
        scheduler.step()
        
    model.eval()

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
    }
    
    return checkpoint


def test(model, dataloader, device=None):
    """Runs inference on the model for the dataloader.

    Args:
        model (pytorch model): model to run
        dataloader (pytorch dataloader): dataloader to evaluate on

    Returns: 
        model's accuracy on the dataloader and a 2D numpy array of
        shape (dataset size, num classes) containing the model's logit outputs
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(device).eval()
    correct = 0.
    total = 0.
    outputs = []
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(images)

        predictions = torch.max(output.data, 1)[1]
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        probabilities = nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        outputs.append(probabilities)  

    acc = correct / total
    outputs = np.vstack(outputs).squeeze()
    
    model.train()
    return acc, outputs
