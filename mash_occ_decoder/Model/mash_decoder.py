from mash_occ_decoder.Model.Decoder.Transformer.v1 import (
    MashDecoder as TSource,
)
from mash_occ_decoder.Model.Decoder.Transformer.v2 import (
    MashDecoder as TPoseEmbed,
)
from mash_occ_decoder.Model.Decoder.Transformer.v3 import (
    MashDecoder as TAllEmbed,
)
from mash_occ_decoder.Model.Decoder.Mamba.v1 import MashDecoder as MSource
from mash_occ_decoder.Model.Decoder.Mamba.v2 import MashDecoder as MPoseEmbed
from mash_occ_decoder.Model.Decoder.Mamba.v3 import MashDecoder as MAllEmbed
from mash_occ_decoder.Model.Decoder.Mamba.v4 import MashDecoder as MPermMash

MashDecoder = MAllEmbed
