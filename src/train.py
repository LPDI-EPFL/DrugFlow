import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml


import sys
basedir = Path(__file__).resolve().parent.parent
sys.path.append(str(basedir))

from src.model.lightning import DrugFlow
from src.model.dpo import DPO
from src.utils import set_deterministic, disable_rdkit_logging, dict_to_namespace, namespace_to_dict


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
            print(f'[CONFIG UPDATE] {key}: {value} -> {config[key]}')
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--backoff', action='store_true')
    p.add_argument('--finetune', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--overfit', action='store_true')
    args = p.parse_args()

    set_deterministic(seed=42)
    disable_rdkit_logging()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config
    assert not (args.resume is not None and args.backoff)
    config['dpo_mode'] = config.get('dpo_mode', None)
    assert not (config['dpo_mode'] and 'checkpoint' not in config), 'DPO mode requires a reference checkpoint'

    if args.debug:
        config['run_name'] = 'debug'

    out_dir = Path(config['train_params']['logdir'], config['run_name'])
    checkpoints_root_dir = Path(out_dir, 'checkpoints')
    if args.backoff:
        last_checkpoint = Path(checkpoints_root_dir, 'last.ckpt')
        print(f'Checking if there is a checkpoint at: {last_checkpoint}')
        if last_checkpoint.exists():
            print(f'Found existing checkpoint: {last_checkpoint}')
            args.resume = str(last_checkpoint)
        else:
            print(f'Did not find {last_checkpoint}')

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None and not args.finetune:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        print(f'Resuming from epoch {ckpt["epoch"]}')
        resume_config = ckpt['hyper_parameters']
        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    if args.debug:
        print('DEBUG MODE')
        args.wandb_params.mode = 'disabled'
        args.train_params.enable_progress_bar = True
        args.train_params.num_workers = 0

    if args.overfit:
        print('OVERFITTING MODE')

    args.eval_params.outdir = out_dir
    model_class = DPO if args.dpo_mode else DrugFlow
    model_args = {
        'pocket_representation': args.pocket_representation,
        'train_params': args.train_params,
        'loss_params': args.loss_params,
        'eval_params': args.eval_params,
        'predictor_params': args.predictor_params,
        'simulation_params': args.simulation_params,
        'virtual_nodes': args.virtual_nodes,
        'flexible': args.flexible,
        'flexible_bb': args.flexible_bb,
        'debug': args.debug,
        'overfit': args.overfit,
    }
    if args.dpo_mode:
        print('DPO MODE')
        model_args.update({
            'dpo_mode': args.dpo_mode,
            'ref_checkpoint_p': args.checkpoint,
        })
    pl_module = model_class(**model_args)

    resume_logging = False
    if args.finetune:
        resume_logging = 'allow'
    elif args.resume is not None:
        resume_logging = 'must'
    
    logger = pl.loggers.WandbLogger(
        save_dir=args.train_params.logdir,
        project='FlexFlow',
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume=resume_logging,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoints_root_dir,
            save_last=True,
            save_on_train_epoch_end=True,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(checkpoints_root_dir, 'val_loss'),
            filename="epoch_{epoch:04d}_loss_{loss/val:.3f}",
            monitor="loss/val",
            save_top_k=5,
            mode="min",
            auto_insert_metric_name=False,
        ),
    ]

    # For learning rate logging
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    default_strategy = 'auto' if pl.__version__ >= '2.0.0' else None
    trainer = pl.Trainer(
        max_epochs=args.train_params.n_epochs,
        logger=logger,
        callbacks=checkpoint_callbacks + [lr_monitor],
        enable_progress_bar=args.train_params.enable_progress_bar,
        check_val_every_n_epoch=args.eval_params.eval_epochs,
        num_sanity_val_steps=args.train_params.num_sanity_val_steps,
        accumulate_grad_batches=args.train_params.accumulate_grad_batches,
        accelerator='gpu' if args.train_params.gpus > 0 else 'cpu',
        devices=args.train_params.gpus if args.train_params.gpus > 0 else 'auto',
        strategy=('ddp_find_unused_parameters_true' if args.train_params.gpus > 1 else default_strategy),
        use_distributed_sampler=False,
    )

    # add all arguments as dictionaries because WandB does not display
    # nested Namespace objects correctly
    logger.experiment.config.update({'as_dict': namespace_to_dict(args)}, allow_val_change=True)

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)

    # # run test set
    # result = trainer.test(ckpt_path='best')
