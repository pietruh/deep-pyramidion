# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from argparse import Namespace
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (FairseqEncoder, FairseqEncoderDecoderModel,
                            register_model, register_model_architecture)
from fairseq.models.transformer import (Embedding, TransformerDecoder,
                                        TransformerModel)
from fairseq.modules import (FairseqDropout, LayerDropModuleList, LayerNorm,
                             PositionalEmbedding,
                             SinusoidalPositionalEmbedding,
                             TransformerEncoderLayer)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


@register_model("pyramidion")
class PyramidionModel(FairseqEncoderDecoderModel):
    """
    Pyramidion model from `"Spa" (Vaswani, et al, 2017)     # TODO: add documentations
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (PyramidionEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Pyramidion model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.Pyramidion_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        # ---- TOPK ARGS ----
        parser.add_argument('--flip-right', type=int, default=1,
                            help='Whether the right side of scores that will be softmaxed against'
                                 'top-k scores should be `mirrored` left-to-right. '
                                 'This should be preferred IMO')
        parser.add_argument('--sort-back', type=int, default=0,
                            help='Whether to sort-back pooled representations at each step.'
                                 ' This used to be 0 for enc-pool-dec architectures, '
                                 'but in hierarchical sorting, there is a need to sort I believe.')
        parser.add_argument('--topk-softmax-base', type=int, default=20,
                            help='The value is a base for exponentiation used in the topk selection'
                                 'mechanism. The higher the value, the closer the output '
                                 'to the real hard topk selection, but this may negatively impact '
                                 'training and the model quality.')
        # ---- POOLER ARGS ----
        parser.add_argument('--no-decoder-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings in decoder')
        parser.add_argument('--global-positions', type=int, default=0,
                            help='add position to each token based on their chunk position')
        parser.add_argument('--encoder-pooling', type=str, default='lambda',
                            choices=['lambda', 'topk'],
                            help='name of the pooling mechanism to use between encoder ' \
                                 'and decoder (no pooling by default)')
        parser.add_argument('--encoder-pooling-arch', type=str, default='linear',
                            choices=['linear', 'ffn'],
                            help='pooling classifier architecture')
        parser.add_argument('--chunk-size', type=int, default=512,
                            help='Size of the chunk, that will be consumed by encoder at once. '
                                 'Note that memory depends quadratically over `chunk size` '
                                 'and linearly over '
                                 'number of chunks one will get after splitting into chunks.')
        parser.add_argument('--enc-layers-and-token-width', type=str,
                            default="{0: 2048, 1: 512}",
                            help='List of two-tuples, where each two-tuple contains layer number '
                                 'and strength of bottleneck to be applied (e.g. (1, 2) means there will'
                                 ' be narrowed down from x to x/2 chunks after the first '
                                 'layer in the encoder).')

        # ---- SPARSE ATTENTION SPECIFIC ----
        parser.add_argument('--use-sparse-attn', type=str, default='pooler',
                            choices=['pooler', 'only_blockwise', 'pooler_no_block'],
                            help='Type of attention to use. '
                                 '`pooler` uses multihead and blockwise,  '
                                 '`only_blockwise` uses ')  # TODO: fix documentation


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        pooler = cls.build_pooler(args)
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, pooler)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_pooler(cls, args):
        from .pooler_utils import TopkPooler
        return TopkPooler(args)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, pooler):
        return PyramidionEncoder(args, src_dict, embed_tokens, pooler)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class PyramidionEncoder(FairseqEncoder):
    """
    Pyramidion encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, pooler):
        self.args = args
        super().__init__(dictionary)
        self.pooler = pooler
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

        self.pooler_pyramid = self.build_pooler_pyramid()

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        original_bs, _ = src_tokens.shape
        src_tokens, src_lengths = self._pad_before_embeddings(src_tokens)

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        if self.pooler.is_lambda:
            # compute padding mask
            encoder_padding_mask = x.eq(0).all(2)

            has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

            # account for padding while computing the representation
            if has_pads:
                x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer_i, layer in enumerate(self.layers):
            if self.pooler.is_lambda:
                x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)
            else:
                x = self.unfold_maybe(x, layer_i)
                x = layer(x, encoder_padding_mask=None)
                x = self.fold_back_maybe(x, original_bs)
                x = self.pooler(x, layer_i=layer_i)

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        encoder_padding_mask = self._recreate_masks(x)
        assert encoder_padding_mask.shape[0] == original_bs
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def _pad_before_embeddings(self, src_tokens: torch.Tensor):
        """Pad tokens to the input length of the first layer. """
        bs, input_len = src_tokens.shape
        cut_to_length = self.pooler_pyramid[0][
            'input_len']  # This is due to the translation task limitations
        src_tokens = src_tokens[..., :cut_to_length]
        src_padded = torch.ones(bs, cut_to_length, dtype=src_tokens.dtype,
                                device=src_tokens.device).fill_(self.dictionary.pad_index)
        src_padded[:, :input_len] = src_tokens
        src_lengths = torch.LongTensor([cut_to_length]*bs, device=src_tokens.device)
        return src_padded, src_lengths

    def unfold_maybe(self, x, layer_i):
        """Some operations to prepare selection-pooling have to be performed before a layer."""
        self.prepare_hierarchical_pooler(layer_i)
        if self.needs_unfold:
            chunk_size = min(x.shape[0], self.args.chunk_size)
            x = x.unfold(0, chunk_size, chunk_size) \
                .permute(1, 0, 3, 2) \
                .flatten(0, 1) \
                .transpose(0, 1)  # Len x Batch x Dim
        return x

    def _recreate_masks(self, x):
        """Recreate False-filled mask for layers after pooling,
        as this is not tractable, and can be assumed without loss of generality"""
        encoder_padding_mask = x.eq(0).all(2).transpose(0, 1)
        return encoder_padding_mask

    def fold_back_maybe(self, encoder_out, old_bs):
        """Documents are already chunked, put them back to the old shape (from before pooling)"""
        if self.needs_unfold:
            # combine chunks back to original batches,
            # but with shorter length (pooled in the 1st dim)
            # this can be achieved by tiling the tensors
            hidden_dim = encoder_out.shape[2]
            enc = encoder_out.shape
            assert enc[1] % old_bs == 0
            assert enc[2] == self.args.encoder_embed_dim

            encoder_out = (
                encoder_out.transpose(0, 1)
                    .reshape([old_bs, -1, hidden_dim])
                    .transpose(0, 1)
            )
        return encoder_out

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def prepare_hierarchical_pooler(self, layer_i):
        """Build pyramid for hierarchical pooler"""
        # set it before pooling for the current layer
        if self.args.enc_layers_and_token_width:
            layer_settings = self.pooler_pyramid[layer_i]
            self.pooler.pooler_config.input_len = layer_settings['input_len']
            self.pooler.pooler_config.pooled_len = layer_settings['pooled_len']
            self.pooler.pooler_config.depth = -1        # topk can derive it on its own.
            self.pooler.pooler_config.sort_back = self.args.sort_back
        else:
            """Default setting is to divide by 2 at each layer(step)"""
            if layer_i == 0:
                # initialize ...
                self.pooler.pooler_config.input_len = self.args.max_source_positions
                self.pooler.pooler_config.pooled_len = self.args.max_source_positions // 2
                self.pooler.pooler_config.sort_back = self.args.sort_back
            else:
                #  ... and reuse the same pooler
                self.pooler.pooler_config.input_len //= 2
                self.pooler.pooler_config.pooled_len //= 2
        # set config to the selector
        self.pooler.selector.set_config(self.pooler.pooler_config)

    def build_pooler_pyramid(self):
        targets = eval(self.args.enc_layers_and_token_width)
        max_layer = len(self.layers) - 1
        assert max(
            targets.keys()) <= max_layer,\
            "Wrongly constructed targets! " \
            "Too many layers mentioned in the targets of 'enc_layers_and_token_width', " \
            "but not enought encoder layers!"
        pyramid = {}
        input_len = self.args.max_source_positions
        pooled_len = input_len
        for lay_i in range(max_layer + 1):
            if lay_i in targets:
                pooled_len = targets[lay_i]
            layer_settings = {'input_len': input_len, 'pooled_len': pooled_len}
            input_len = pooled_len
            pyramid[lay_i] = layer_settings
        for k, v in pyramid.items():
            print(f'Layer {k} :  {v}')
        return pyramid

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return self.embed_positions.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    @property
    def needs_unfold(self):
        types_where_blockwise_needed = ['pooler', 'only_blockwise']
        return self.args.use_sparse_attn in types_where_blockwise_needed


@register_model_architecture('pyramidion', 'pyramidion_base')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
