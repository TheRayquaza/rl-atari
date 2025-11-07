import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from pipeline import RLTrainer

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point"""
    
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    trainer = RLTrainer(cfg)
    
    if cfg.training.enabled:
        trainer.train()
        trainer.create_plots()
    
    if cfg.evaluation.enabled:
        trainer.evaluate()
    
    log.info(f"Results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
