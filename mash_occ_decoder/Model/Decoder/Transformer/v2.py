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
        mask_degree: int = 4,
        sh_degree: int = 3,
        depth: int = 24,
        hidden_dim: int = 400,
        hidden_embed_dim: int = 48,
        output_dim=1,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        super().__init__()

        self.depth = depth

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        self.anchor_embed = PointEmbed(
            6, hidden_embed_dim, hidden_dim - self.mask_dim - self.sh_dim
        )
        self.point_embed = PointEmbed(3, hidden_embed_dim, hidden_dim)

        def get_latent_attn():
            return PreNorm(
                hidden_dim,
                Attention(
                    hidden_dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1
                ),
            )

        def get_latent_ff():
            return PreNorm(hidden_dim, FeedForward(hidden_dim, drop_path_rate=0.1))

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
            hidden_dim,
            Attention(hidden_dim, hidden_dim, heads=1, dim_head=hidden_dim),
            context_dim=hidden_dim,
        )
        self.decoder_ff = PreNorm(hidden_dim, FeedForward(hidden_dim))

        self.to_outputs = nn.Linear(hidden_dim, output_dim)
        return

    def embedMash(self, mash_params: torch.Tensor) -> torch.Tensor:
        anchor_embeddings = self.anchor_embed(mash_params[:, :, :6])

        mash_embeddings = torch.cat(
            [anchor_embeddings, mash_params[:, :, 6:]],
            dim=2,
        )
        return mash_embeddings

    def forward(self, data_dict):
        mash_params = data_dict["mash_params"]
        queries = data_dict["qry"]

        x = self.embedMash(mash_params)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        latents = latents + self.decoder_ff(latents)

        occ = self.to_outputs(latents).squeeze(-1)
        return occ
