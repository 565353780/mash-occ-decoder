import torch
from torch import nn
from functools import partial

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mash_occ_decoder.Model.mamba_block import create_block, init_weights
from mash_occ_decoder.Model.Layer.pre_norm import PreNorm
from mash_occ_decoder.Model.Layer.feed_forward import FeedForward
from mash_occ_decoder.Model.Layer.attention import Attention
from mash_occ_decoder.Model.Layer.point_embed import PointEmbed


class MashDecoder(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
        d_hidden: int = 400,
        d_hidden_embed: int = 100,
        n_layer: int = 96,
        n_cross: int = 4,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device="cuda:0",
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        assert d_hidden % 4 == 0

        self.rotation_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.position_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.mask_embed = PointEmbed(self.mask_dim, d_hidden_embed, d_hidden // 4)
        self.sh_embed = PointEmbed(self.sh_dim, d_hidden_embed, d_hidden // 4)
        self.point_embed = PointEmbed(3, d_hidden_embed, d_hidden)

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_hidden, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_hidden,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.decoder_cross_attn = PreNorm(
            d_hidden,
            Attention(d_hidden, d_hidden, heads=n_cross, dim_head=d_hidden),
            context_dim=d_hidden,
        )
        self.decoder_ff = PreNorm(d_hidden, FeedForward(d_hidden))

        self.to_outputs = nn.Linear(d_hidden, 1)
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

        for layer in self.layers:
            x, residual = layer(x)
            x = x + residual

        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        latents = latents + self.decoder_ff(latents)

        occ = self.to_outputs(latents).squeeze(-1)
        return occ
