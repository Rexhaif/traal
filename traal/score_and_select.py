import utils
import numpy as np
import omegaconf as omg
from rich.logging import RichHandler
import transformers as tr
from datasets import Dataset
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)


def add_full_selection(dataset: Dataset) -> Dataset:
    if 'selected' in dataset.features:
        dataset = dataset.remove_columns(['selected'])

    selected = [1] * len(dataset)
    return dataset.add_column('selected', selected)

def add_random_selection(dataset: Dataset, n_select: int = 50) -> Dataset:

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


def add_selection_from_score(dataset: Dataset, column: str, n_select: int = 50) -> Dataset:
    if "selected" not in dataset.features:
        selection_mark = np.zeros(len(dataset)).astype(bool)

        indices = np.array(range(len(dataset)), dtype=int)
    else:
        selection_mark = np.array(dataset['selected']).astype(bool)
        dataset = dataset.remove_columns(['selected'])
        indices = np.array(range(len(dataset)), dtype=int)
        indices = indices[~selection_mark]

    scores = np.array(dataset[column])[indices] # getting score values for considered indices
    indices = indices[np.argsort(scores)[::-1]][:n_select] # selecting top indices by score values
    selection_mark[indices] = 1
    dataset = dataset.add_column('selected', selection_mark)
    return dataset


def add_scores(dataset: Dataset, config: omg.DictConfig, score_column: str) -> Dataset:
    if "selected" not in dataset.features:
        # fresh start, no need to run model, just add random score for random seed dataset
        scores = np.random.sample(size=len(dataset))
    else:
        # not a fresh start, actually doing something
        directory = utils.make_paths(
            config.dataset.kind,
            config.experiment.name,
            config.experiment.seed,
            config.experiment.time,
            config.experiment.id
        )['model']

        

def run_score_and_select(config: omg.DictConfig):
    dataset = utils.load_dataset(config.dataset.uri)

    al_kind = config.experiment.kind

    if al_kind == "full":
        logger.info("===> Data Selection Mode = Full, selecting entire training set")
        dataset['train'] = add_full_selection(dataset['train'])
    elif al_kind == "al-random":
        logger.info(f"===> Data Selection Mode = AL with random, selecting {config.dataset.sample_size} random examples")
        dataset['train'] = add_random_selection(dataset['train'], n_select=config.dataset.sample_size)
    elif al_kind == "al-score":
        logger.info(f"===> Data Selection Mode = AL with computed scores, running score computation and selcting from scores")
        dataset['train'] = add_scores(dataset['train'], config, config.dataset.score_column)
        dataset['train'] = add_selection_from_score(dataset['train'], config.dataset.score_column, n_select=config.dataset.sample_size)

    logger.info(f"Saving dataset to {config.dataset.uri}")
    utils.save_dataset(config.dataset.uri, dataset)

