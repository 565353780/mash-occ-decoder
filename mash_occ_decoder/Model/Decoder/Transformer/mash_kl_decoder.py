import torch
from torch import nn

from mash_occ_decoder.Model.Layer.pre_norm import PreNorm
from mash_occ_decoder.Model.Layer.feed_forward import FeedForward
from mash_occ_decoder.Model.Layer.attention import Attention
from mash_occ_decoder.Model.Layer.point_embed import PointEmbed
from mash_occ_decoder.Method.cache import cache_fn
from mash_occ_decoder.Module.diagonal_gaussian_distribution import DiagonalGaussianDistribution


class MashKLDecoder(nn.Module):
    def __init__(
        self,
        depth=24,
        mash_dim: int = 25,
        dim: int = 512,
        queries_dim=512,
        output_dim=1,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
    ):
        super().__init__()

        self.depth = depth

        self.point_embed = PointEmbed(dim=queries_dim)

        self.proj = nn.Linear(mash_dim, dim)

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
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim))

        self.to_outputs = nn.Linear(queries_dim, output_dim)

        self.mean_fc = nn.Linear(dim, dim)
        self.logvar_fc = nn.Linear(dim, dim)
        return

    def forward(self, data_dict: dict):
        mash_params = data_dict["mash_params"]
        queries = data_dict["qry"]
        drop_prob = data_dict['drop_prob']
        deterministic = data_dict['deterministic']

        if drop_prob > 0.0:
            mask = mash_params.new_empty(*mash_params.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            mash_params = mash_params * mask.unsqueeze(-1).expand_as(mash_params).type(mash_params.dtype)

        mash_params = self.proj(mash_params)

        mean = self.mean_fc(mash_params)
        logvar = self.logvar_fc(mash_params)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        mash_params = posterior.sample()
        kl = posterior.kl()

        for self_attn, self_ff in self.layers:
            mash_params = self_attn(mash_params) + mash_params
            mash_params = self_ff(mash_params) + mash_params

        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=mash_params)

        latents = latents + self.decoder_ff(latents)

        occ = self.to_outputs(latents).squeeze(-1)

        occ = torch.sigmoid(occ)

        result_dict = {
            'occ': occ,
            'kl': kl,
        }

        return result_dict
