import csv
import pandas as pd
import os
import random

LABEL_CSV = r'G:\Data_Resources\2022_2_data\train_label\train_label.csv'
OUTDIR = r'G:\Data_Resources\2022_2_data\train_label'
train_size = 0.7

def readcsv(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', sep='\t', header=None)
    df = df.values.tolist()

    return df

def writecsv(data, csv_path):
    df = pd.DataFrame(data)
    df.to_csv(csv_path, header=0, index=0, sep='\t')

def train_val_split(all_data, train_size=0.7):
    train_data = []
    val_data = []
    for sample in all_data:
        if random.random() < train_size:
            train_data.append(sample)
        else:
            val_data.append(sample)

    return train_data, val_data

if __name__ == "__main__":
    all_data = readcsv(LABEL_CSV)
    train_data, val_data = train_val_split(all_data, train_size=train_size)
    writecsv(train_data, os.path.join(OUTDIR, 'train.csv'))
    writecsv(val_data, os.path.join(OUTDIR, 'val.csv'))