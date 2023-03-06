from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import os
import urllib.request
import zipfile
import datasets

TASKPATH = {
    "CoLA": 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
    "SST": 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    "MRPC": 'https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv',
    "QQP": 'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
    "STS": 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
    "MNLI": 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
    "QNLI": 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
    "RTE": 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
    "WNLI": 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
    "diagnostic": 'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'
}


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


def get_torch_dataset(tokenizer_name_or_path, task_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    dataset = datasets.load_dataset('glue', task_name)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.map(lambda x: tokenizer(x['sentence'], truncation=True, max_length=128), batched=True)

    columns = ['input_ids', 'attention_mask', 'labels']
    dataset.set_format(type='torch', columns=columns)

    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    return train_dataset, eval_dataset, None



