import os
from multiprocessing import Pool, Queue
import omegaconf as omg
from typing import List
import numpy as np
import logging
import utils
from functools import partial
from rich.logging import RichHandler
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

CUDA_DEVICES = Queue()
WORKER_CUDA_DEVICES = None


def initialize_worker(num_cuda_devices=1):
    global CUDA_DEVICES
    global WORKER_CUDA_DEVICES
    WORKER_CUDA_DEVICES = []
    for i in range(num_cuda_devices):
        WORKER_CUDA_DEVICES.append(str(CUDA_DEVICES.get()))
    logger.info(f"Worker cuda devices: {','.join(WORKER_CUDA_DEVICES)}")


def common_task_fn(args):
    kind = args['kind']
    if kind == "full":
        one_run_task_fn(args['args'])
    elif kind == "al":
        al_iterations_task_fn(args['args'])

def one_run_task_fn(task):
    script, name, base_conf, seed, id_, additional_args = task
    time_str = utils.get_time_for_saving()
    if WORKER_CUDA_DEVICES is None or len(WORKER_CUDA_DEVICES) == 0:
        cuda_str = 'CUDA_VISIBLE_DEVICES=""'
        logger.warn(f"Running without cuda devices, worker cuda devices empty or not set {WORKER_CUDA_DEVICES}")
    else:
        cuda_str = f'CUDA_VISIBLE_DEVICES={",".join(WORKER_CUDA_DEVICES)}'

    task_str = f"{cuda_str} python {script} base_conf={base_conf} " \
        f"experiment.seed={seed} experiment.id={id_} experiment.time={time_str} " \
        f"experiment.name={name}+{seed} experiment.group={name} {additional_args}"

    time.sleep(np.random.randint(1_000) / 1_000)
    logger.info(f"Running command: {task_str}")
    ret = os.system(task_str)
    ret = str(ret)
    logger.info(f'Task "{name}+{seed}" finished with return code: {ret}.')

def al_iterations_task_fn(task):
    script, name, base_conf, seed, id_, additional_args, n_iterations, dataset_path = task
    time_str = utils.get_time_for_saving()
    if WORKER_CUDA_DEVICES is None or len(WORKER_CUDA_DEVICES) == 0:
        cuda_str = 'CUDA_VISIBLE_DEVICES=""'
        logger.warn(f"Running without cuda devices, worker cuda devices empty or not set {WORKER_CUDA_DEVICES}")
    else:
        cuda_str = f'CUDA_VISIBLE_DEVICES={",".join(WORKER_CUDA_DEVICES)}'

    

    for i in range(n_iterations):
        logger.info(f"===> AL Iteration {i+1}")
        task_str = f"{cuda_str} python {script} base_conf={base_conf} " \
            f"experiment.seed={seed} experiment.id={id_} experiment.time={time_str} " \
            f"experiment.name={name}+{seed} experiment.group={name} dataset.uri={dataset_path} " \
            f"experiment.iteration={i+1} {additional_args}"
        ret = os.system(task_str)
        ret = str(ret)
        logger.info(f'Task "{name}+{seed}" iteration {i+1} finished with return code: {ret}.')
        assert ret == "0", "Task failed, stopping experiment"


def run_tasks(config: omg.DictConfig):
    for idx in config.cuda_devices:
        CUDA_DEVICES.put(idx)

    logger.info(f"Starting {len(config.cuda_devices)} workers")

    tasks = []
    for task in config.tasks:
        if task.kind == "full":
            for seed in task.seeds:
                id_ = utils.create_experiment_id()
                tasks.append({
                    "kind": "full",
                    "args": (task.script, task.name, task.base_conf, seed, id_, task.additional_args)
                })
        if task.kind == "al":
            for seed in task.seeds:
                id_ = utils.create_experiment_id()
                new_dataset_path = f"s3://traal-storage/experiments/{task.name}-{seed}"
                logger.info(f"Copying dataset from {task.dataset} to {new_dataset_path}")
                utils.copy_dataset(task.dataset, new_dataset_path)
                tasks.append({
                    "kind": "al",
                    "args": (task.script, task.name, task.base_conf, seed, id_, task.additional_args, task.n_iterations, new_dataset_path)
                })

    logger.info(f"Running {len(config.tasks)} experiments, {len(tasks)} run total(with seeds)")

    pool = Pool(
        processes=len(config.cuda_devices),
        initializer=partial(initialize_worker, num_cuda_devices=1),
    )
    try:
        pool.map(common_task_fn, tasks)

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    cli_args = omg.OmegaConf.from_cli()

    main_conf = omg.OmegaConf.load(cli_args.conf)

    config = omg.OmegaConf.merge(main_conf, cli_args)

    run_tasks(config)


    


