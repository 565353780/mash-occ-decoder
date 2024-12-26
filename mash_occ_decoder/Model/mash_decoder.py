from mash_occ_decoder.Model.Decoder.Transformer.v1 import (
    MashDecoder as TSource,
)
from mash_occ_decoder.Model.Decoder.Transformer.v2 import (
    MashDecoder as TAllEmbed,
)
from mash_occ_decoder.Model.Decoder.Transformer.v3 import (
    MashDecoder as TVD,
)

from mash_occ_decoder.Model.Decoder.Mamba.v1 import MashDecoder as MSource
from mash_occ_decoder.Model.Decoder.Mamba.v2 import MashDecoder as MAllEmbed

MashDecoder = TVD
