import torch
from torch import nn

from mash_occ_decoder.Model.Layer.pre_norm import PreNorm
from mash_occ_decoder.Model.Layer.feed_forward import FeedForward
from mash_occ_decoder.Model.Layer.attention import Attention
from mash_occ_decoder.Model.Layer.point_embed import PointEmbed
from mash_occ_decoder.Method.cache import cache_fn


class SHDecoder(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
        depth: int = 12,
        hidden_dim: int = 100,
        hidden_embed_dim: int = 100,
        query_dim: int = 100,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
    ):
        super().__init__()

        self.depth = depth

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        self.rotation_embed = PointEmbed(3, hidden_embed_dim, hidden_dim // 2)
        self.position_embed = PointEmbed(3, hidden_embed_dim, hidden_dim // 2)

        def get_latent_attn():
            return PreNorm(
                query_dim,
                Attention(
                    query_dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1
                ),
            )

        def get_latent_ff():
            return PreNorm(query_dim, FeedForward(query_dim, drop_path_rate=0.1))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_ff = PreNorm(query_dim, FeedForward(query_dim))

        self.to_outputs = nn.Linear(query_dim, self.mask_dim + self.sh_dim)
        return

    def embedPose(self, mash_params: torch.Tensor) -> torch.Tensor:
        rotation_embeddings = self.rotation_embed(mash_params[:, :, :3])
        position_embeddings = self.position_embed(mash_params[:, :, 3:6])

        pose_embeddings = torch.cat([rotation_embeddings, position_embeddings], dim=2)
        return pose_embeddings

    def forward(self, pose_params: torch.Tensor) -> torch.Tensor:
        x = self.embedPose(pose_params)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        latents = x + self.decoder_ff(x)

        shape_params = self.to_outputs(latents)
        return shape_params
