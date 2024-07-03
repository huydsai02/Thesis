import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config, Wav2Vec2FeatureProjection, Wav2Vec2Adapter, 
    Wav2Vec2PositionalConvEmbedding, _compute_mask_indices, Wav2Vec2FeatureEncoder, 
    Wav2Vec2PreTrainedModel, Wav2Vec2FeedForward, Wav2Vec2AttnAdapterLayer
)
from collections import OrderedDict
from transformers.models.deprecated.transfo_xl.modeling_transfo_xl import TransfoXLModelOutput
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (CausalLMOutput, Wav2Vec2BaseModelOutput, BaseModelOutput, ModelOutput)
from typing import Optional, Tuple, Union, Dict, List
import warnings
import numpy as np
from dataclasses import dataclass
# warnings.filterwarnings("ignore")

_HIDDEN_STATES_START_POSITION = 2

@dataclass
class CausalLMOutputWithMems(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mems: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class Wav2Vec2BaseModelOutputWithMems(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mems: Optional[Tuple[torch.FloatTensor, ...]] = None

class Wav2Vec2PositionalCausalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
        )
        print(config.hidden_size, config.hidden_size, config.num_conv_pos_embeddings, config.num_conv_pos_embedding_groups)
        self.pad = (config.num_conv_pos_embeddings - 1, 0)
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(F.pad(hidden_states, self.pad))
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
    

class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Wav2Vec2Config] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        mems: torch.Tensor = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
#         print(mems)
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            if mems is not None:
                # print(mems)
                hidden_states = torch.cat([mems, hidden_states], dim=1)    
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if attention_mask is not None:
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(attention_mask, mask_value)
#             print("MASK VALUE:", mask_value)
            
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
#         print("ATTN WEIGHT:")
#         print(attn_weights)
#         print("=" * 100)
        
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
    

class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, mems=None, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, mems=mems, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor, mems=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        if mems is not None:
            mems = self.layer_norm(mems)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, mems=mems, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config, type_positional_conv_embedding, mem_length, num_frames, use_streaming_attention_mask):
        super().__init__()
        self.config = config
        self.mem_length = mem_length
        self.num_frames = num_frames
        self.use_streaming_attention_mask = use_streaming_attention_mask

        print("NUM FRAMES:", self.num_frames)
        print("MEM LENGTH:", self.mem_length)
        print("TYPE POSITIONAL CON EMBEDDING:", type_positional_conv_embedding.upper())
        print("USE STREAMING ATTENTION MASK:", self.use_streaming_attention_mask)
        
        self.type_positional_conv_embedding = type_positional_conv_embedding
        if type_positional_conv_embedding == "standard":
            self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        elif type_positional_conv_embedding == "causal":
            self.pos_conv_embed = Wav2Vec2PositionalCausalConvEmbedding(config)
        else:
            assert False, "Need type positional conv embedding"
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
    def _update_mems(self, mems, hidden_states):
        if self.mem_length <= 0:
            return None
    
        if mems is None:
            mems = hidden_states
        else:
            assert len(mems) == len(hidden_states), "len(hids) != len(mems)"
            with torch.no_grad():
                for key in hidden_states:
                    mems[key] = torch.cat([mems[key], hidden_states[key]], dim=1)
        for key in mems:
            mems[key] = mems[key][:,-self.mem_length:,:].detach()

        return mems
    
    def forward(
        self,
        hidden_states: torch.tensor,
        mems: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = ()
        all_self_attentions = () if output_attentions else None
        hids = {"inp": hidden_states}
        
        if self.use_streaming_attention_mask:
            bsz = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            fd = bsz * self.config.num_attention_heads
            attention_mask = torch.zeros((fd, seq_len, seq_len), device=hidden_states.device)
            nf, mlen = self.num_frames, self.mem_length
            for i in range(0, seq_len, nf):
                attention_mask[:, i:i+nf, max(i-mlen, 0):i+nf] = 1
            attention_mask = attention_mask != 1
#         if attention_mask is not None:
#             # make sure padded tokens output 0
#             expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
#             hidden_states[~expand_attention_mask] = 0

#             # extend attention_mask
#             attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
#             attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
#             attention_mask = attention_mask.expand(
#                 attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
#             )
        if self.training:
            assert mems is None, "mems is None during training phrase"
            
        if mems is None or self.type_positional_conv_embedding == "standard":
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
        else:
            qlen = hidden_states.shape[1]
            merge_inp = torch.cat([mems["inp"], hidden_states], dim=1)
            position_embeddings = self.pos_conv_embed(merge_inp)
            merge_inp = merge_inp + position_embeddings
            hidden_states = merge_inp[:,-qlen:,:]
        
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for i, layer in enumerate(self.layers):
            hids[i] = hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)
            mem_i = mems[i] if mems is not None else None
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states, mem_i,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, mem_i, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        new_mems = self._update_mems(mems, hids)
        
        all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, new_mems, all_hidden_states, all_self_attentions] if v is not None)
        
        return TransfoXLModelOutput(
            last_hidden_state=hidden_states,
            mems=new_mems,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config, type_positional_conv_embedding, mem_length, num_frames, use_streaming_attention_mask):
        super().__init__()
        self.config = config
        self.mem_length = mem_length
        self.num_frames = num_frames
        self.use_streaming_attention_mask = use_streaming_attention_mask

        print("NUM FRAMES:", self.num_frames)
        print("MEM LENGTH:", self.mem_length)
        print("TYPE POSITIONAL CON EMBEDDING:", type_positional_conv_embedding.upper())
        print("USE STREAMING ATTENTION MASK:", self.use_streaming_attention_mask)
        
        self.type_positional_conv_embedding = type_positional_conv_embedding
        if type_positional_conv_embedding == "standard":
            self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        elif type_positional_conv_embedding == "causal":
            self.pos_conv_embed = Wav2Vec2PositionalCausalConvEmbedding(config)
        else:
            assert False, "Need type positional conv embedding"
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def _update_mems(self, mems, hidden_states):
        if self.mem_length <= 0:
            return None
    
        if mems is None:
            mems = hidden_states
        else:
            assert len(mems) == len(hidden_states), "len(hids) != len(mems)"
            with torch.no_grad():
                for key in hidden_states:
                    mems[key] = torch.cat([mems[key], hidden_states[key]], dim=1)
        for key in mems:
            mems[key] = mems[key][:,-self.mem_length:,:].detach()

        return mems
    
    def forward(
        self,
        hidden_states,
        mems: Optional[torch.Tensor] = None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = ()
        all_self_attentions = () if output_attentions else None
        hids = {"inp": hidden_states}

#         if attention_mask is not None:
#             # make sure padded tokens are not attended to
#             expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
#             hidden_states[~expand_attention_mask] = 0

#             # extend attention_mask
#             attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
#             attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
#             attention_mask = attention_mask.expand(
#                 attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
#             )
        
        if self.use_streaming_attention_mask:
            bsz = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            fd = bsz * self.config.num_attention_heads
            attention_mask = torch.zeros((fd, seq_len, seq_len), device=hidden_states.device)
            nf, mlen = self.num_frames, self.mem_length
            for i in range(0, seq_len, nf):
                attention_mask[:, i:i+nf, max(i-mlen, 0):i+nf] = 1
            attention_mask = attention_mask != 1
                    
        if self.training:
            assert mems is None, "mems is None during training phrase"
            
        if mems is None or self.type_positional_conv_embedding == "standard":
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
        else:
            qlen = hidden_states.shape[1]
            merge_inp = torch.cat([mems["inp"], hidden_states], dim=1)
            position_embeddings = self.pos_conv_embed(merge_inp)
            merge_inp = merge_inp + position_embeddings
            hidden_states = merge_inp[:,-qlen:,:]

        hidden_states = self.dropout(hidden_states)
        
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for i, layer in enumerate(self.layers):
            hids[i] = hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)
            mem_i = mems[i] if mems is not None else None
            
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states, mem_i,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, mem_i, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]
                
            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        new_mems = self._update_mems(mems, hids)
        hidden_states = self.layer_norm(hidden_states)
        all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, new_mems, all_hidden_states, all_self_attentions] if v is not None)
        
        return TransfoXLModelOutput(
            last_hidden_state=hidden_states,
            mems=new_mems,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
    
class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, type_positional_conv_embedding, mem_length, num_frames, use_streaming_attention_mask):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config, type_positional_conv_embedding, mem_length, num_frames, use_streaming_attention_mask)
        else:
            self.encoder = Wav2Vec2Encoder(config, type_positional_conv_embedding, mem_length, num_frames, use_streaming_attention_mask)
#         self.NUM_FRAME = 32 # 645ms
        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        mems: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )
   
        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            mems=mems,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        all_hidden_states = encoder_outputs.hidden_states
        
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        return Wav2Vec2BaseModelOutputWithMems(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=all_hidden_states,
            attentions=encoder_outputs.attentions,
            mems=encoder_outputs.mems
        )
    

STEPS_PER_LOG = 1000
class Wav2Vec2SmallNonStreamingForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config, need_feature_transform=False, 
                 loss_log_path=None, mse_hidden_states_ratio = 0.3,
                 mse_logits_ratio = 1, indices_hidden_learn = None,
                 use_one_reshape_layer = False, mem_length = 192, 
                 type_positional_conv_embedding = "causal", #standard
                 target_lang: Optional[str] = None,
                 use_streaming_attention_mask = False,
                 num_frames = 32
                ):

        super().__init__(config)

        self.TEACHER_HIDDEN_SIZE = 1024

        if indices_hidden_learn != None:
            nhl = config.num_hidden_layers
            assert len(indices_hidden_learn) in [nhl, nhl + 1]
            self.indices_hidden_learn = sorted(indices_hidden_learn)

        assert mse_hidden_states_ratio >= 0 and mse_hidden_states_ratio <= 1
        assert mse_logits_ratio >= 0 and mse_logits_ratio <= 1

        print("NUM HIDDEN LAYERS:", config.num_hidden_layers)
        print("HIDDEN SIZE:", config.hidden_size)
        print("FEATURE TRANSFORM:", need_feature_transform)
        print("LOSS LOG PATH:", loss_log_path)
        print("MSE HIDDEN STATES RATIO:", mse_hidden_states_ratio)
        print("MSE LOGITS RATIO:", mse_logits_ratio)
        print("USE ONE RESHAPE LAYER:", use_one_reshape_layer)
        # assert False
        
        self.use_one_reshape_layer = use_one_reshape_layer
        self.need_feature_transform = need_feature_transform
        self.mse_hidden_states_ratio = mse_hidden_states_ratio
        self.mse_logits_ratio = mse_logits_ratio

        # assert False
        self.wav2vec2 = Wav2Vec2Model(
            config, type_positional_conv_embedding, 
            mem_length, num_frames, use_streaming_attention_mask
        )
        
        self.dropout = nn.Dropout(config.final_dropout)
        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        if config.hidden_size != self.TEACHER_HIDDEN_SIZE and self.mse_hidden_states_ratio > 0:
            if self.use_one_reshape_layer:
                self.linear_reshape = nn.Linear(config.hidden_size, self.TEACHER_HIDDEN_SIZE, bias=False)
            else:
                self.linear_reshape = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.TEACHER_HIDDEN_SIZE, bias=False)
                    for _ in range(len(indices_hidden_learn))
                ])

        if self.need_feature_transform:
            #### CODE NVLB
            self.feature_transform = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(config.hidden_size, config.hidden_size)),
                ('bn1', nn.BatchNorm1d(config.hidden_size)),
                ('activation1', nn.LeakyReLU()),
                ('drop1', nn.Dropout(config.final_dropout)),
                ('linear2', nn.Linear(config.hidden_size, config.hidden_size)),
                ('bn2', nn.BatchNorm1d(config.hidden_size)),
                ('activation2', nn.LeakyReLU()),
                ('drop2', nn.Dropout(config.final_dropout)),
                ('linear3', nn.Linear(config.hidden_size, config.hidden_size)),
                ('bn3', nn.BatchNorm1d(config.hidden_size)),
                ('activation3', nn.LeakyReLU()),
                ('drop3', nn.Dropout(config.final_dropout))
            ]))
        #######################

        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.teacher_generate_func = None
        if loss_log_path is not None:
            self.log_file = open(loss_log_path, "a")

        self.mse_pred_losses, self.mse_hs_losses, self.ctc_losses, self.mix_losses = [], [], [], []
        # Initialize weights and apply final processing
        self.post_init()
        
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def set_teacher_generate_func(self, func):
        self.teacher_generate_func = func
        
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        mems: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_input_values = None
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if teacher_input_values is not None and self.teacher_generate_func is not None:
            teacher_logits, teacher_hidden_states = self.teacher_generate_func(teacher_input_values)
            
        outputs = self.wav2vec2(
            input_values,
            mems=mems,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        if self.need_feature_transform:
            ##### CODE NVLB ###
            B, T, F = hidden_states.size()
            hidden_states = hidden_states.view(B * T, F)

            hidden_states = self.feature_transform(hidden_states)

            hidden_states = hidden_states.view(B, T, F)
            ###################

        logits = self.lm_head(hidden_states)
        # print(logits)
        loss = None
        if labels is not None:
            mse_logits = mse_hidden_states = torch.tensor(0)
            
            mse_logits = nn.functional.mse_loss(
                nn.functional.softmax(logits, dim=-1), 
                nn.functional.softmax(teacher_logits, dim=-1),
                reduction='none'
            ).sum(dim=-1).mean()
            
            if self.mse_hidden_states_ratio > 0:
                hidden_states_reshape = outputs["hidden_states"]

                if self.config.hidden_size != self.TEACHER_HIDDEN_SIZE:
                    if self.use_one_reshape_layer:
                        hidden_states_stacked = torch.stack(hidden_states_reshape)
                        hidden_states_stacked = self.linear_reshape(hidden_states_stacked)
                    else:
                        hidden_states_reshape = hidden_states_reshape[-len(self.linear_reshape):]
                        hidden_states_reshape = [
                            reshape_layer(hidden_states_reshape[i]) for i, reshape_layer in enumerate(self.linear_reshape)
                        ]
                        hidden_states_stacked = torch.stack(hidden_states_reshape)
                else:
                    hidden_states_stacked = torch.stack(hidden_states_reshape)
                    
                teacher_hs_stacked = torch.stack([
                    teacher_hidden_states[i] for i in self.indices_hidden_learn
                ])
                
                mse_hidden_states = nn.functional.mse_loss(
                    hidden_states_stacked,
                    teacher_hs_stacked,
                    reduction='none'
                )

                mse_hidden_states = mse_hidden_states.view(len(self.indices_hidden_learn), -1).mean(dim=-1).sum()

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            alpha = 1 - self.mse_hidden_states_ratio
            beta = 1 - self.mse_logits_ratio

            ctc_loss = loss
            loss = (1 - alpha) * mse_hidden_states + alpha * (beta * ctc_loss + (1 - beta) * mse_logits)
            self.log_file.write(f"MSE logits = {mse_logits.item()}, MSE hidden states = {mse_hidden_states.item()}, CTC loss = {ctc_loss.item()}, Mix = {loss.item()}\n")
            
            self.ctc_losses.append(ctc_loss.item())
            self.mix_losses.append(loss.item())
            self.mse_pred_losses.append(mse_logits.item())
            self.mse_hs_losses.append(mse_hidden_states.item())
            
            if (len(self.ctc_losses) - 1) % STEPS_PER_LOG == 0:
                avg_mse_pred = np.mean(self.mse_pred_losses[-STEPS_PER_LOG:])
                avg_mse_hs = np.mean(self.mse_hs_losses[-STEPS_PER_LOG:])
                avg_ctc = np.mean(self.ctc_losses[-STEPS_PER_LOG:])
                avg_mix = np.mean(self.mix_losses[-STEPS_PER_LOG:])
                step = len(self.ctc_losses)
                print(f"Step {step}: MSE logits = {avg_mse_pred}, MSE hidden states = {avg_mse_hs}, CTC loss = {avg_ctc}, Mix = {avg_mix}\n")
   
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithMems(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, mems=outputs.mems
        )