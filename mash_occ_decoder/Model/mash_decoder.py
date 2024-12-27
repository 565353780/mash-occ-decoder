from mash_occ_decoder.Model.Decoder.Transformer.mash_decoder import MashDecoder as TM
from mash_occ_decoder.Model.Decoder.Transformer.latent_kl_decoder import LatentKLDecoder as TL

from mash_occ_decoder.Model.Decoder.Mamba.v1 import MashDecoder as MSource
from mash_occ_decoder.Model.Decoder.Mamba.v2 import MashDecoder as MAllEmbed

MashDecoder = TM
