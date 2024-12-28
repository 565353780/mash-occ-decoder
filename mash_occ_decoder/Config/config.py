class MashDecoderConfig(object):
    def __init__(self) -> None:
        # Marching Cube realted
        self.mc_chunk_size = 3000
        self.mc_res0 = 128
        self.mc_up_steps = 2
        self.mc_threshold = 0.5
        return


MASH_DECODER_CONFIG = MashDecoderConfig()
