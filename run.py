from pathlib import Path
import argparse
import datetime
import torch
import lightning as pl
from lightning.pytorch import loggers as plg
from lightning.pytorch.callbacks import ModelCheckpoint
from configs.config import return_defaults

from src.models.model import FullModel

torch.set_float32_matmul_precision('high')
pl.seed_everything(42)

def main():
    parser = argparse.ArgumentParser(description='GraphGL Network')
    cfg = return_defaults()
    list_of_args = list(dict(cfg).keys())
    list_of_args.remove('debug')

    for k in list_of_args: parser.add_argument('--config', default='misc/standard') if k == 'config' else parser.add_argument(f'--{k}')
    
    parser.add_argument('--resume-training', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = vars(parser.parse_args())
    dictlist = []
    for key, value in list(args.items()): 
        if value is not None: dictlist.append(key), dictlist.append(value) 

    cfg.merge_from_file(f'{cfg.path}/configs/{args["config"]}.yaml')
    cfg.merge_from_list(dictlist)
    cfg.freeze()
    
    ckpt_dir = Path(f'{cfg.path}/weights/checkpoints/{args["config"]}')
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    t = datetime.datetime.now()
    time_string = "_".join(str(t.time()).split(":")[:3])
    name = f'{cfg.name}_{time_string}'
    filename = f"{cfg.config.split('/')[-1]}"
    
    if not cfg.debug: 
        wandb_logger = plg.WandbLogger(entity="bev-cv-project", project="GraphGL", save_dir=f'{cfg.path}/logs/', name=f'{cfg.config}', version=name, log_model=False, config=dict(cfg))
        checkpoint_callback = ModelCheckpoint(monitor="val_epoch_loss", mode="min", dirpath=ckpt_dir, save_top_k=1, filename=filename,)

    callbacks = [checkpoint_callback] if not cfg.debug else []
    loggers = [wandb_logger] if not cfg.debug else []
    devices = [0] if cfg.debug else -1

    model = FullModel(args=cfg)
    
    if cfg.resume_training:
        ckpt = ckpt_dir.glob('*.ckpt')
        ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime)[-1]
        
    trainer = pl.Trainer(max_epochs=cfg.epochs, accelerator='gpu', 
                        devices=devices,
                        logger=loggers, 
                        callbacks=callbacks,
                        check_val_every_n_epoch=cfg.acc_interval, num_sanity_val_steps=0,
                        overfit_batches=10 if cfg.debug else 0,
                        )

    trainer.fit(model, ckpt_path=ckpt if cfg.resume_training else None)
    trainer.test(model)



if __name__ == '__main__':
    main()
