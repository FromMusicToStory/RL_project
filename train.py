from argparse import ArgumentParser
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from model import DQNClassification


def main(config_path):
    config = yaml.safe_load(open(config_path))

    model = DQNClassification(config, run_mode='train')
    model.train()
    lr_monitor = pl.callbacks.LearningRateMonitor()
    checkponiter = pl.callbacks.ModelCheckpoint(dirpath=config['checkpoint_path'],
                                                filename='model_{epoch:02d}-{train_loss:.2f}',
                                                verbose=True, save_last=True, save_top_k=3, monitor='train_loss',
                                                mode='min')
    # mode is max because train_loss will be the reward
    trainer = pl.Trainer(
        accelerator='gpu', gpus=config['ngpu'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        log_every_n_steps=config['log_every_n_steps'],
        max_epochs=config['max_epochs'],
        logger=WandbLogger(config['project']),
        callbacks=[lr_monitor, checkponiter])
    trainer.fit(model)


if __name__ == "__main__":
    main("config.yaml")
