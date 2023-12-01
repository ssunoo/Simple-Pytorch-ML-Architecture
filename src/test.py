import torch
import os
import argparse
import pandas as pd
import json
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import LinearRegressionModel, Config, Processor
from dataset import CustomDataset

def mseError(df, pred_col: str, label_col: str):
    err = 0
    L = len(df)
    if(L == 0):
        return 0.0
    
    for i in range(L):
        err += (df.loc[i][pred_col] - df.loc[i][label_col]) ** 2
    err /= len(df)
    return err

def test(model, batch_size, output_path, test_dataset):
    test_set = DataLoader(test_dataset, batch_size, shuffle=False)
    model.eval()
    preds = []
    labels = []

    for (raw_data, y) in tqdm(test_set):
        input = processor(raw_data)
        with torch.no_grad():
            pred = model(**input)            
            pred = pred.cpu().view(-1).tolist()
            y = y.view(-1).tolist()
            preds += pred
            labels += y

    result_df = pd.DataFrame({'predict': preds, 'label': labels})
    result_df.to_csv(output_path, index=False)

    err = mseError(result_df, 'predict', 'label')

    return err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--model_config', type=str, default="")
    args = parser.parse_args()

    if args.device >= 0:
        torch_device = "cuda:{0}".format(args.device)
    else:
        torch_device = "cpu"

    with open(args.model_config) as json_file:
        config = Config(**json.load(json_file))
    
    model = LinearRegressionModel(config).to(torch_device)
    processor = Processor()
    pth = torch.load(args.model_path)
    model.load_state_dict(pth)

    test_dataset = CustomDataset()

    model_name = "_".join(args.model_path.split('/')[-2:])
    result_dir = os.path.join(args.output_dir, model_name)
    
    try:
        os.makedirs(result_dir)
    except:
        pass

    err = test(model, 1, os.path.join(result_dir, "test_output.csv"), test_dataset)
    summary_df = pd.DataFrame({'model name': model_name, 'mse error': [err]})

    print(summary_df)
    summary_path = os.path.join(result_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
