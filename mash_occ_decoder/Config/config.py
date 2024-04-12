class MashDecoderConfig(object):
    def __init__(self) -> None:
        # dataset related
        self.dir_data = "/home/chli/Dataset/aro_net/data"
        self.name_dataset = "shapenet"
        self.n_wk = 4
        self.categories_train = ["02691156", "03001627"]
        self.categories_test = ["02691156", "03001627"]
        # MashDecoder hyper-parameters
        self.n_qry = 200
        # common hyper-parameters
        self.n_bs = 500
        # Marching Cube realted
        self.mc_chunk_size = 3000
        self.mc_res0 = 64
        self.mc_up_steps = 2
        self.mc_threshold = 0.5
        return


MASH_DECODER_CONFIG = MashDecoderConfig()
