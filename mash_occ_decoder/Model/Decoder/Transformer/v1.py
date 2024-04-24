import torch
from torch import nn

from mash_occ_decoder.Model.Layer.pre_norm import PreNorm
from mash_occ_decoder.Model.Layer.feed_forward import FeedForward
from mash_occ_decoder.Model.Layer.attention import Attention
from mash_occ_decoder.Model.Layer.point_embed import PointEmbed
from mash_occ_decoder.Method.cache import cache_fn


class MashDecoder(nn.Module):
    def __init__(
        self,
        depth=24,
        dim=31,
        queries_dim=400,
        output_dim=1,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        super().__init__()

        self.depth = depth

        self.point_embed = PointEmbed(dim=queries_dim)

        def get_latent_attn():
            return PreNorm(
                dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
            )

        def get_latent_ff():
            return PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=queries_dim),
            context_dim=dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim))

        self.to_outputs = nn.Linear(queries_dim, output_dim)
        return

    def forward(self, data_dict):
        x = data_dict["mash_params"]
        queries = data_dict["qry"]

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        latents = latents + self.decoder_ff(latents)

        occ = self.to_outputs(latents).squeeze(-1)
        return occ