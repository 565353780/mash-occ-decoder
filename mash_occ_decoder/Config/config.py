class MashDecoderConfig(object):
    def __init__(self) -> None:
        # dataset related
        self.dir_data = "/home/chli/Dataset/aro_net/data"
        self.name_dataset = "shapenet"
        self.name_single = "fertility"
        self.n_wk = 4
        self.categories_train = ["02691156"]
        self.categories_test = ["02691156", "03001627"]
        # MashDecoder hyper-parameters
        self.n_anc = 40
        self.n_qry = 100000
        # common hyper-parameters
        self.device = "cuda"
        self.mode = "train"
        self.n_bs = 36
        self.n_epochs = 10000
        self.lr = 1e-5
        self.freq_decay = 10
        self.weight_decay = 0.999
        # Marching Cube realted
        self.mc_chunk_size = 3000
        self.mc_res0 = 64
        self.mc_up_steps = 2
        self.mc_threshold = 0.5

        assert self.name_dataset in ["abc", "shapenet", "single", "custom"]
        assert self.mode in ["train", "test"]
        return


MASH_DECODER_CONFIG = MashDecoderConfig()
