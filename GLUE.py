# pip install datasets
# pip install transformers
from datasets import load_dataset
import numpy as np
from datasets import Dataset 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile

import io
TASKPATH = {"CoLA":'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
             "SST":'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
             "MRPC":'https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv',
             "QQP":'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
             "STS":'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
             "MNLI":'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
             "QNLI":'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
             "RTE":'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
             "WNLI":'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
             "diagnostic":'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'}

def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    if task == "MNLI":
        print("\tNote (12/10/20): This script no longer downloads SNLI. You will need to manually download and format the data to use SNLI.")
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASKPATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


def data_loader():
    ds = load_dataset("mariosasko/glue", "cola") #adjust to the uploaded file or use this
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    ds = ds.map(lambda example: tokenizer(example["sentence"]), batched=True)
    ds.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    return ds

if __name__ == '__main__':
    ds=data_loader()
    Train=ds["train"]
    Test=ds["test"]
    Validation=ds["validation"]