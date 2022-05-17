from asyncio.log import logger
from datasets import DatasetDict, Dataset
from datasets.filesystems import S3FileSystem
from uuid import uuid4
import json
from datetime import datetime
import transformers as tr
import numpy as np
import os
from typing import Dict, Any
from pathlib import Path
import shutil


NER_TAG_MAPPING = {
    'conll2003': ['O', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER']
}


def add_to_json_log(log_dict: Dict[str, Any], json_file: Path, key: str = 'acquisition'):
    if not json_file.exists():
        base_dict = {
            'acquisition': [],
            'successor': []
        }
    else:
        with json_file.open('r', encoding='utf-8') as f:
            base_dict = json.load(f)

    base_dict[key].append(log_dict)

    with json_file.open('w', encoding="utf-8") as f:
        json.dump(base_dict, f, ensure_ascii=False, default=str)

def load_dataset(dataset_uri: str):
    if "s3://" in dataset_uri:
        s3fs = S3FileSystem()
        return DatasetDict.load_from_disk(dataset_uri, fs=s3fs)
    else:
        return DatasetDict.load_from_disk(dataset_uri)

def save_dataset(dataset_uri: str, dataset: DatasetDict):
    if "s3://" in dataset_uri:
        s3fs = S3FileSystem()
        dataset.save_to_disk(dataset_uri, fs=s3fs)
    else:
        dataset.save_to_disk(dataset_uri)

def count_n_tokens(dataset: Dataset) -> int:
    counter = 0
    for row in dataset:
        counter += len(row['tokens'])

    return counter

def copy_dataset(source_uri: str, destination_uri: str):
    if 's3://' in source_uri or "s3://" in destination_uri:
        ret = os.system(f"aws s3 cp --recursive {source_uri} {destination_uri}")
    else:
        shutil.copytree(source_uri, destination_uri, dirs_exist_ok=True)


def add_random_selection(dataset: Dataset, n_select: int = 50):

    if "selected" not in dataset.features:
        selection_mark = np.zeros(len(dataset)).astype(bool)

        indices = np.array(range(len(dataset)), dtype=int)
    else:
        selection_mark = np.array(dataset['selected']).astype(bool)
        dataset = dataset.remove_columns(['selected'])
        indices = np.array(range(len(dataset)), dtype=int)
        indices = indices[~selection_mark]
        
    indices = np.random.choice(indices, size=n_select)
    selection_mark[indices] = 1

    dataset = dataset.add_column('selected', selection_mark)
    return dataset


def select_examples(dataset):
    selected = dataset.filter(lambda x: x['selected'] == 1, batched=False)
    return selected


def get_checkpoint_dir(base_dir: str, best_trial: tr.trainer_utils.BestRun):
    best_run_dir = Path(base_dir, f"run-{best_trial.run_id}")
    max_step = -1
    max_step_path = None
    for dir in best_run_dir.iterdir():
        step_n = int(dir.parts[-1].split("-")[1])
        if step_n > max_step:
            max_step = step_n
            max_step_path = dir

    return max_step_path


def get_time_for_saving():
    current_datatime = datetime.now()
    return current_datatime.strftime("%m-%d-%a-%H-%M")


def create_experiment_id() -> str:
    base_uuid = str(uuid4()).split("-")[-1]
    return base_uuid