import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml

import sys
basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from src.size_predictor.size_model import SizeModel
from src.utils import set_deterministic, disable_rdkit_logging


def dict_to_namespace(input_dict):
    """ Recursively convert a nested dictionary into a Namespace object """
    if isinstance(input_dict, dict):
        output_namespace = Namespace()
        output = output_namespace.__dict__
        for key, value in input_dict.items():
            output[key] = dict_to_namespace(value)
        return output_namespace

    elif isinstance(input_dict, Namespace):
        # recurse as Namespace might contain dictionaries
        return dict_to_namespace(input_dict.__dict__)

    else:
        return input_dict


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        # if isinstance(value, dict):
        #     arg_dict[key] = Namespace(**value)
        # else:
        #     arg_dict[key] = value
        arg_dict[key] = dict_to_namespace(value)

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__

        if isinstance(value, dict):
            # update dictionaries recursively
            value = merge_configs(config[key], value)

        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    set_deterministic(seed=42)
    disable_rdkit_logging()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']
        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    if args.debug:
        print('DEBUG MODE')
        args.run_name = 'debug'
        args.wandb_params.mode = 'disabled'
        args.train_params.enable_progress_bar = True
        args.train_params.num_workers = 0
        # torch.manual_seed(1234)

    out_dir = Path(args.train_params.logdir, args.run_name)
    # args.eval_params.outdir = out_dir
    pl_module = SizeModel(
        max_size=args.max_size,
        pocket_representation=args.pocket_representation,
        train_params=args.train_params,
        loss_params=args.loss_params,
        eval_params=None, #args.eval_params,
        predictor_params=args.predictor_params,
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.train_params.logdir,
        project='FlexFlow',
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume='must' if args.resume is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(out_dir, 'checkpoints'),
            filename="best-acc={accuracy/val:.2f}-epoch={epoch:02d}",
            monitor="accuracy/val",
            save_top_k=1,
            save_last=True,
            mode="max",
            # save_on_train_epoch_end=True,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(out_dir, 'checkpoints'),
            filename="best-mse={mse/val:.2f}-epoch={epoch:02d}",
            monitor="loss/train",
            save_top_k=1,
            save_last=False,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.train_params.n_epochs,
        logger=logger,
        callbacks=checkpoint_callbacks,
        enable_progress_bar=args.train_params.enable_progress_bar,
        # check_val_every_n_epoch=args.eval_params.eval_epochs,
        num_sanity_val_steps=args.train_params.num_sanity_val_steps,
        accumulate_grad_batches=args.train_params.accumulate_grad_batches,
        accelerator='gpu' if args.train_params.gpus > 0 else 'cpu',
        devices=args.train_params.gpus if args.train_params.gpus > 0 else 'auto',
        strategy=('ddp' if args.train_params.gpus > 1 else None)
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)

    # # run test set
    # result = trainer.test(ckpt_path='best')
