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
        d_anchor: int = 31,
        d_hidden: int = 400,
        n_layer: int = 24,
        n_cross: int = 8,
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

        self.anchor_embed = PointEmbed(6, 48, d_hidden - d_anchor + 6)
        self.point_embed = PointEmbed(3, 48, d_hidden)

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

    def forward(self, data_dict):
        mash_params = data_dict["mash_params"]
        queries = data_dict["qry"]

        anchor_embeddings = self.anchor_embed(mash_params[:, :, :6])

        x = torch.cat([anchor_embeddings, mash_params[:, :, 6:]], dim=2)

        for layer in self.layers:
            x, residual = layer(x)
            x = x + residual

        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        latents = latents + self.decoder_ff(latents)

        occ = self.to_outputs(latents).squeeze(-1)
        return occ
