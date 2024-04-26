from mash_occ_decoder.Model.Decoder.Transformer.v1 import (
    MashDecoder as TSource,
)
from mash_occ_decoder.Model.Decoder.Transformer.v2 import (
    MashDecoder as TAllEmbed,
)
try:
    from mash_occ_decoder.Model.Decoder.Mamba.v1 import MashDecoder as MSource
    from mash_occ_decoder.Model.Decoder.Mamba.v2 import MashDecoder as MAllEmbed
except:
    print('[ERROR][mash_decoder::import]')
    print('\t import mamba model failed! current env can only run with transformer models')

MashDecoder = TAllEmbed
