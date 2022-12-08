import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import hydra
from omegaconf import DictConfig,OmegaConf
from model import DQNClassification

@hydra.main(version_base=None, config_path='conf', config_name='dqn')
def main(config: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(config)}')
    model = DQNClassification(config.model, run_mode='train')
    model.train()
    lr_monitor = pl.callbacks.LearningRateMonitor()
    checkponiter = pl.callbacks.ModelCheckpoint(dirpath=config.checkpoint_path,
                                                filename='model_{epoch:02d}-{train_loss:.2f}',
                                                verbose=True, save_last=True, save_top_k=3, monitor='train_loss',
                                                mode='min', save_on_train_epoch_end=True)
    # mode is max because train_loss will be the reward
    trainer = pl.Trainer(
        accelerator='gpu', devices=config.gpu,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        max_epochs=config.max_epochs,
        logger=WandbLogger(project=config.project, name=config.name),
        callbacks=[lr_monitor, checkponiter])
    trainer.fit(model)


if __name__ == "__main__":
    main()

