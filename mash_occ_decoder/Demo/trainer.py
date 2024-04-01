from mash_occ_decoder.Module.trainer import Trainer


def demo():
    model_file_path = None

    trainer = Trainer(model_file_path)

    trainer.train()
    return True
