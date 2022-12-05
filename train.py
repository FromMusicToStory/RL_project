from argparse import ArgumentParser
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from model import RLClassification


def main(config_path):
    config = yaml.safe_load(open(config_path))

    model = RLClassification(config, run_mode='train')
    model.train()
    lr_monitor = pl.callbacks.LearningRateMonitor()
    checkponiter = pl.callbacks.ModelCheckpoint(dir_path=config['checkpoint_path'],
                                                filename='{epoch:02d}-{train_loss:.2f}',
                                                verbose=True, save_last=True, monitor='train_loss',
                                                mode='max', prefix="model_")
    # mode is max because train_loss will be the reward
    trainer = pl.Trainer(
        accelerator='gpu', gpus=config['gpus'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        log_every_n_steps=config['log_every_n_steps'],
        max_epochs=config['max_epochs'],
        logger=WandbLogger(config['project']),
        callbacks=[lr_monitor, checkponiter])
    trainer.fit(model)


if __name__ == "__main__":
    main("config.yaml")
