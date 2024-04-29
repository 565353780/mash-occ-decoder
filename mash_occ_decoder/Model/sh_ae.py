import torch
from torch import nn

from mash_occ_decoder.Model.Layer.pre_norm import PreNorm
from mash_occ_decoder.Model.Layer.feed_forward import FeedForward
from mash_occ_decoder.Model.Layer.attention import Attention
from mash_occ_decoder.Model.Layer.point_embed import PointEmbed
from mash_occ_decoder.Method.cache import cache_fn


class SHAutoEncoder(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
        depth: int = 24,
        hidden_dim: int = 400,
        hidden_embed_dim: int = 100,
        encode_dim: int = 4,
        decode_cross_heads: int = 4,
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

        self.mask_embed = PointEmbed(self.mask_dim, hidden_embed_dim, encode_dim // 2)
        self.sh_embed = PointEmbed(self.sh_dim, hidden_embed_dim, encode_dim // 2)

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    encode_dim,
                    Attention(encode_dim, hidden_dim, heads=1, dim_head=hidden_dim),
                    context_dim=hidden_dim,
                ),
                PreNorm(encode_dim, FeedForward(encode_dim)),
            ]
        )

        def get_latent_attn():
            return PreNorm(
                encode_dim,
                Attention(
                    encode_dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1
                ),
            )

        def get_latent_ff():
            return PreNorm(encode_dim, FeedForward(encode_dim, drop_path_rate=0.1))

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
            encode_dim,
            Attention(
                encode_dim, hidden_dim, heads=decode_cross_heads, dim_head=hidden_dim
            ),
            context_dim=hidden_dim,
        )
        self.decoder_ff = PreNorm(encode_dim, FeedForward(encode_dim))

        self.to_outputs = nn.Linear(encode_dim, self.mask_dim + self.sh_dim)
        return

    def embedPose(self, mash_params: torch.Tensor) -> torch.Tensor:
        rotation_embeddings = self.rotation_embed(mash_params[:, :, :3])
        position_embeddings = self.position_embed(mash_params[:, :, 3:6])

        pose_embeddings = torch.cat([rotation_embeddings, position_embeddings], dim=2)
        return pose_embeddings

    def embedShape(self, mash_params: torch.Tensor) -> torch.Tensor:
        mask_embeddings = self.mask_embed(mash_params[:, :, 6 : 6 + self.mask_dim])
        sh_embeddings = self.sh_embed(mash_params[:, :, 6 + self.mask_dim :])

        shape_embeddings = torch.cat(
            [mask_embeddings, sh_embeddings],
            dim=2,
        )
        return shape_embeddings

    def encodeShape(self, mash_params: torch.Tensor) -> torch.Tensor:
        pose_embeddings = self.embedPose(mash_params)

        shape_embeddings = self.embedShape(mash_params)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(shape_embeddings, context=pose_embeddings, mask=None)
            + shape_embeddings
        )
        x = cross_ff(x) + x
        return x

    def decodeShape(self, x: torch.Tensor, pose_params: torch.Tensor) -> torch.Tensor:
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        pose_embeddings = self.embedPose(pose_params)
        latents = self.decoder_cross_attn(x, context=pose_embeddings)

        latents = latents + self.decoder_ff(latents)

        shape_params = self.to_outputs(latents)
        return shape_params

    def forward(self, mash_params: torch.Tensor) -> torch.Tensor:
        x = self.encodeShape(mash_params)
        shape_params = self.decodeShape(x, mash_params[:, :, :6])
        return shape_params
