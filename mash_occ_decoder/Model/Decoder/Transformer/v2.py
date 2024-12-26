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
        mask_degree: int = 3,
        sh_degree: int = 2,
        depth: int = 24,
        hidden_dim: int = 400,
        hidden_embed_dim: int = 100,
        hidden_cross_heads: int = 4,
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

        assert hidden_dim % 4 == 0

        self.rotation_embed = PointEmbed(3, hidden_embed_dim, hidden_dim // 4)
        self.position_embed = PointEmbed(3, hidden_embed_dim, hidden_dim // 4)
        self.mask_embed = PointEmbed(self.mask_dim, hidden_embed_dim, hidden_dim // 4)
        self.sh_embed = PointEmbed(self.sh_dim, hidden_embed_dim, hidden_dim // 4)
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
            Attention(
                hidden_dim, hidden_dim, heads=hidden_cross_heads, dim_head=hidden_dim
            ),
            context_dim=hidden_dim,
        )
        self.decoder_ff = PreNorm(hidden_dim, FeedForward(hidden_dim))

        self.to_outputs = nn.Linear(hidden_dim, output_dim)
        return

    def embedMash(self, mash_params: torch.Tensor) -> torch.Tensor:
        rotation_embeddings = self.rotation_embed(mash_params[:, :, :3])
        position_embeddings = self.position_embed(mash_params[:, :, 3:6])
        mask_embeddings = self.mask_embed(mash_params[:, :, 6 : 6 + self.mask_dim])
        sh_embeddings = self.sh_embed(mash_params[:, :, 6 + self.mask_dim :])

        mash_embeddings = torch.cat(
            [rotation_embeddings, position_embeddings, mask_embeddings, sh_embeddings],
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
