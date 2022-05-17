from datasets import Dataset, DatasetDict
import utils
import omegaconf as omg
import wandb
from metrics import SeqEvalMetrics
from typing import List, Tuple
import transformers as tr
import shutil
from rich.logging import RichHandler
import argparse as ap
from pathlib import Path
import optuna
import os
import time
import logging
import score_and_select

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

def prepare_dataset(
    config: omg.DictConfig,
    tokenizer: tr.PreTrainedTokenizer, 
) -> Tuple[DatasetDict, List[int]]:

    logger.info("==> Running scoring and seletion")
    score_and_select.run_score_and_select(config)

    dataset = utils.load_dataset(config.dataset.uri)

    id2label = utils.NER_TAG_MAPPING[config.dataset.kind]
    label2id = {k:i for i, k in enumerate(id2label)}

    # filtering out unselected examples
    dataset['train'] = utils.select_examples(dataset['train'])

    logger.info(f"===> Labeled Dataset size: {len(dataset['train'])}")
    logger.info(f"===> Full dataset has following structure: {dataset}")

    def convert_label_to_ids(item):
        """
        Converting str tags to int tags according to label2id
        """

        return {
            'ner': [label2id[x] for x in item['ner']]
        }

    logger.info("Remapping ner tags str -> idx")
    dataset = dataset.map(convert_label_to_ids, batched=False, num_proc=config.dataset.n_jobs)

    def tokenize_and_align_labels(examples):
        """
        Aligning ner tags with tokenization
        """
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    logger.info("Tokenization and tag alignment")
    dataset = dataset.map(tokenize_and_align_labels, batched=True, batch_size=128, num_proc=config.dataset.n_jobs)

    return dataset, id2label


def run_al_training(config: omg.DictConfig):

    t_start = time.time()

    tr.trainer_utils.set_seed(config.experiment.seed)
    logger.info("===> Initializing WandB")
    paths = utils.make_paths(
        config.dataset.kind,
        config.experiment.name,
        config.experiment.seed,
        config.experiment.time,
        config.experiment.id
    )
    
    wandb_client = wandb.init(
        project=config.wandb.project,
        dir=str(paths['base']),
        group=config.experiment.group,
        name=config.experiment.name,
        entity=config.wandb.entity,
        tags=[config.dataset.kind, config.experiment.time],
        resume="allow",
        id=config.experiment.id
    )

    tokenizer = tr.AutoTokenizer.from_pretrained(config.model.name)

    full_dataset, id2label = prepare_dataset(
        config,
        tokenizer=tokenizer
    )

    metrics = SeqEvalMetrics(id2label=id2label)

    def model_init():
        return tr.AutoModelForTokenClassification.from_pretrained(
            config.model.name, num_labels=len(id2label)
        )

    data_collator = tr.DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = tr.TrainingArguments(
        output_dir=paths['hp_search'],
        evaluation_strategy="epoch",
        save_strategy='epoch',
        disable_tqdm=True,
        group_by_length=True,
        seed=config.experiment.seed,
        fp16=True, fp16_opt_level="O2",
        metric_for_best_model='eval_f1',
        load_best_model_at_end=True,
        report_to='none'
    )
    logger.info(f"Assebled model config: {training_args}")
    logger.info("Initializing trainer")
    trainer = tr.Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=full_dataset['train'],
        eval_dataset=full_dataset["validation"],
        data_collator=data_collator,
        model_init=model_init,
        compute_metrics=metrics.compute_metrics,
    )

    def custom_search_objective(trial: optuna.Trial):
        possible_batch_sizes = [4, 8, 16, 32, 64]
        possible_batch_sizes = [
            x for x in possible_batch_sizes 
            if (len(full_dataset['train']) / x) >= config.model.min_updates_per_epoch
        ]
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3, step=0.01),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ['linear', 'constant', 'cosine']),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", possible_batch_sizes)
        }

    def compute_objective(metrics):
        return metrics['eval_f1']

    logger.info(f"===> Running HP Search with {config.model.n_trials} trials")
    best_trial = trainer.hyperparameter_search(
        hp_space=custom_search_objective,
        compute_objective=compute_objective,
        direction="maximize",
        backend="optuna",
        n_trials=config.model.n_trials # number of trials
    )
    t_end = time.time()
    logger.info(f"===> HP Search done. Best Trial: {best_trial}")
        
    model_dir = utils.get_checkpoint_dir(paths['hp_search'], best_trial)
    save_dir = paths['model']
    if save_dir.exists():
        shutil.rmtree(save_dir)

    logger.info(f"Movig best checkpoint to experiment folder")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_dir.replace(save_dir)

    logger.info("Cleaning up hp search checkpoints")
    shutil.rmtree(paths['hp_search'])

    log_dict = {
        'acquisition/f1': best_trial.objective,
        'acquisition/iteration_time': t_end - t_start,
        'acquisition/total_tokens_labeled': utils.count_n_tokens(full_dataset['train'])
    }

    for key, val in best_trial.hyperparameters.items():
        log_dict[f"acquisition/hp/{key}"] = val

    logger.info(f"Logging to wand {log_dict}")
    wandb_client.log(log_dict)

    logger.info("Saving dict to json log")
    utils.add_to_json_log(log_dict=log_dict, json_file=paths['logs'], key="acquisition")

    wandb_client.finish()
    logger.info("===> WandB finished")

if __name__ == "__main__":
    cli_args = omg.OmegaConf.from_cli()

    base_args = omg.OmegaConf.load(cli_args.base_conf)

    config = omg.OmegaConf.merge(base_args, cli_args)

    logger.info(f"===> Common loaded config:\n{omg.OmegaConf.to_yaml(config)}")
    run_al_training(config)
