import torch
import os
import argparse
import csv
import math
import json
import pandas as pd
import torch.nn as nn
from model import LinearRegressionModel, Config, Processor
from dataset import CustomDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def train(model, processor, train_set, dev_set, n_epochs, batch_size, learning_rate, acc_steps, eval_steps, save_path):    
    tr_set = DataLoader(train_set, batch_size, shuffle=True)

    criterion = model.loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    min_loss = 1e9
    
    for epoch in tqdm(range(n_epochs)):   
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for i, (raw_data, y) in enumerate(tqdm(tr_set, leave=False)):
            input = processor(raw_data)
            pred = model(**input)
            y = y.to("cpu" if pred.get_device() < 0 else pred.get_device())
            loss = criterion(pred, y)
            loss = loss / acc_steps
            total_loss += loss.cpu().item() * batch_size * acc_steps
            loss.backward()
            if (i + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (i + 1) % eval_steps == 0:
                eval_loss = evaluation(model, processor, dev_set)

                if min_loss > eval_loss:
                    min_loss = eval_loss
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                tqdm.write("Validation Loss: {}".format(eval_loss))
                model.train()

        avg_loss = total_loss / len(tr_set.dataset)
        tqdm.write("Epoch{} training loss: {}".format(epoch, avg_loss))
        optimizer.step()

def evaluation(model, processor, dev_set):
    tqdm.write("Evaluating...")
    dev_set = DataLoader(dev_set, 1, shuffle=False)    
    model.eval()
    total_loss = 0

    criterion = model.loss
    for (raw_data, y) in tqdm(dev_set, leave=False):
        input = processor(raw_data)
        with torch.no_grad():
            pred = model(**input)            
            y = y.to("cpu" if pred.get_device() < 0 else pred.get_device())
            loss = criterion(pred, y)
        total_loss += loss.cpu().item()
        avg_loss = total_loss / len(dev_set.dataset)
    return avg_loss   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--evaluation_steps', type=int, default=100)
    parser.add_argument('--model_save_dir', type=str)   
    parser.add_argument('--model_config', type=str)     
    parser.add_argument('--model', type=str, default="")
    args = parser.parse_args()

    if args.device >= 0:
        torch_device = "cuda:{0}".format(args.device)
    else:
        torch_device = "cpu"

    with open(args.model_config) as json_file:
        config = Config(**json.load(json_file))
        
    model = LinearRegressionModel(config).to(torch_device)
    processor = Processor()

    if args.model != "":
        pth = torch.load(args.model)
        model.load_state_dict(pth)

    dev_dataset = CustomDataset()
    train_dataset = CustomDataset()
    train(model, processor, train_dataset, dev_dataset, args.n_epochs, args.batch_size, args.learning_rate, args.accumulation_steps, args.evaluation_steps, args.model_save_dir)