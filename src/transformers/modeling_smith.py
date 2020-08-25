# coding=utf-8
# Copyright 2018 XXX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Smith model. """

def masked_sentence_block_loss(predicted_embeddings, label_embeddings):
    """Computes the masked sentence block loss"""

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sim = torch.tensordot(predicted_embeddings.unsqueeze(1), label_embeddings.T.unsqueeze(0), dims=2)
    B = sim.shape[0]
	
    mask = torch.eye(B).long().to(dev)

    p = torch.exp(sim)/torch.sum(torch.exp(sim), 1)

    log_likelihood = torch.log(p)*mask

    negative_log_likelihood = -torch.sum(log_likelihood)/B

    return negative_log_likelihood
	
def mask_sentence_blocks(sentence_block_batch, num_masked_blocks, split_idx, sentence_block_mask_vector):
    """Randomly masks sentence block vectors

    Args:
        sentence_block_batch: list, each entry is a tensor of sentence block embeddings
        num_masked_blocks: int, number of blocks to mask in each document
        split_idx: list, each entry indicates the number of sentence blocks in an individual document
        sentence_block_mask_vector: tensor, randomly initialized vector to represent masked sentence blocks
    Returns:
        sentence_block_batch: tensor, batch of sentence blocks where num_masked_blocks have been masked for each document
        sentence_block_labels: tensor, true value for the masked sentence blocks
        mask_indices: list, each entry is the index of a masked sentence block
    """

    sentence_block_labels = []
    mask_indices = []
    num_sentence_blocks = max(split_idx)

    for i in range(len(sentence_block_batch)):
        mask_idx = torch.randperm(split_idx[i])[:num_masked_blocks]
        sentence_block_labels.append(sentence_block_batch[i][mask_idx, :])
        sentence_block_batch[i][mask_idx, :] = sentence_block_mask_vector
        mask_indices.extend(mask_idx + num_sentence_blocks*i)
		
    sentence_block_batch = torch.stack(sentence_block_batch)

    sentence_block_labels = torch.cat(sentence_block_labels)

    return sentence_block_batch, sentence_block_labels, mask_indices

import logging
import os
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as f

from .configuration_smith import SmithConfig
from .activations import gelu, gelu_new, swish
from .file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
from .modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "SmithConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

####################################################
# This list contrains shortcut names for some of
# the pretrained weights provided with the models
####################################################
SMITH_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "smith-base-uncased",
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_smith(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer and array have mismatched shapes {pointer.shape} and {array.shape}"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (itself a sub-class of torch.nn.Module)
####################################################

####################################################
# Here is an example of typical layer in a PyTorch model of the library
# The classes are usually identical to the TF 2.0 ones without the 'TF' prefix.
#
# See the conversion methods in modeling_tf_pytorch_utils.py for more details
####################################################

class SmithSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class SmithPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = SmithLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SmithLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = SmithPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class SmithOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SmithLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
		
class SmithSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = SmithLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class SmithAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SmithSelfAttention(config)
        self.output = SmithSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SmithIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SmithOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = SmithLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
		
####################################################
# PreTrainedModel is a sub-class of torch.nn.Module
# which take care of loading and saving pretrained weights
# and various common utilities.
#
# Here you just need to specify a few (self-explanatory)
# pointers for your model and the weights initialization
# method if its not fully covered by PreTrainedModel's default method
####################################################

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
	
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

SmithLayerNorm = torch.nn.LayerNorm

class SmithEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = SmithLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SmithLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SmithAttention(config)
        self.intermediate = SmithIntermediate(config)
        self.output = SmithOutput(config)

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        head_mask=None, 
        output_attentions=False,
    ):
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class SmithEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SmithLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class SmithPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output_norm = f.normalize(pooled_output, dim=1, p=2)
        return pooled_output_norm
		


class SmithPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = SmithConfig
    load_tf_weights = load_tf_weights_in_smith
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, SmithLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Config():
    None

def parse_config(config):
    """Splits input configuration file into separate sentence and document encoder configurations"""
    sentence_config = Config()
    sentence_config.hidden_size = config.hidden_size1
    sentence_config.num_hidden_layers = config.num_hidden_layers1
    sentence_config.num_attention_heads = config.num_attention_heads1
    sentence_config.attention_probs_dropout_prob = config.attention_probs_dropout_prob
    sentence_config.gradient_checkpointing = config.gradient_checkpointing
    sentence_config.vocab_size = config.vocab_size
    sentence_config.type_vocab_size = config.type_vocab_size
    sentence_config.model_type = "Smith"
    sentence_config.max_position_embeddings = config.block_length
    sentence_config.layer_norm_epsilon = config.layer_norm_epsilon
    sentence_config.initializer_range = config.initializer_range
    sentence_config.hidden_dropout_prob = config.hidden_dropout_prob
    sentence_config.hidden_act = config.hidden_act
    sentence_config.intermediate_size = config.intermediate_size
    sentence_config.pad_token_id = config.pad_token_id

    document_config = Config()
    document_config.hidden_size = config.hidden_size2
    document_config.num_hidden_layers = config.num_hidden_layers2
    document_config.num_attention_heads = config.num_attention_heads2
    document_config.attention_probs_dropout_prob = config.attention_probs_dropout_prob
    document_config.gradient_checkpointing = config.gradient_checkpointing
    document_config.vocab_size = config.vocab_size
    document_config.type_vocab_size = config.type_vocab_size
    document_config.model_type = "Smith"
    document_config.max_position_embeddings = config.max_blocks
    document_config.layer_norm_epsilon = config.layer_norm_epsilon
    document_config.initializer_range = config.initializer_range
    document_config.hidden_dropout_prob = config.hidden_dropout_prob
    document_config.hidden_act = config.hidden_act 
    document_config.intermediate_size = config.intermediate_size

    return sentence_config, document_config


SMITH_START_DOCSTRING = r"""    The Smith model was proposed in
    `Beyond 512 Tokens: Siamese Multi-depth Transformer-based Hierarchical Encoder for Document Matching
    <https://arxiv.org/abs/2004.12297>`__ by....
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Parameters:
        config (:class:`~transformers.XxxConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

SMITH_INPUTS_DOCSTRING = r"""
    Inputs:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.XxxTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


class SmithForPreTraining(SmithPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        sentence_config, document_config = parse_config(self.config)

        self.embeddings = SmithEmbeddings(sentence_config)
        self.sentence_encoder = SmithEncoder(sentence_config)
        self.lm_head = SmithOnlyMLMHead(sentence_config)
        
        self.sentence_pooler = SmithPooler(sentence_config)

        self.document_encoder = SmithEncoder(document_config)

        self.sentence_config = sentence_config
        self.document_config = document_config

        self.sentence_block_mask_vector = torch.rand(self.config.hidden_size2)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        split_idx=None,
        labels=None,
        label_docs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        encoder_extended_attention_mask = None
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.sentence_config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs1 = self.sentence_encoder(
            embedding_output, 
            extended_attention_mask, 
            head_mask=head_mask, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output1 = encoder_outputs1[0]
        pooled_output1 = self.sentence_pooler(sequence_output1)

        prediction_scores = self.lm_head(sequence_output1)

        # resplit batch into invididual documents
        pooled_output_split = torch.split(pooled_output1, split_idx, dim=0)
	
        num_sentence_blocks = max(split_idx)

        doc_input_list = []

        # pad each document with zeros so that all documents have the same shape
        for i in range(len(split_idx)):
            num_rows_to_pad = num_sentence_blocks - split_idx[i]
            zero_padding =  torch.zeros(num_rows_to_pad, self.sentence_config.hidden_size, device=device)
            doc_input_list.append(torch.cat([pooled_output_split[i], zero_padding], dim=0))
		
        sentence_block_embeddings, sentence_block_labels, mask_indices = mask_sentence_blocks(
            doc_input_list, self.document_config.max_position_embeddings, split_idx, self.sentence_block_mask_vector.to(device)
        )

        attention_mask = torch.ones_like(sentence_block_embeddings)
        attention_mask[sentence_block_embeddings==0] = 0
        attention_mask = torch.sum(attention_mask, dim=2)
        attention_mask[attention_mask > 0] = 1

        input_shape = sentence_block_embeddings.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask=None, num_hidden_layers=self.document_config.num_hidden_layers)

        encoder_outputs2 = self.document_encoder(
            sentence_block_embeddings, 
            extended_attention_mask, 
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output2 = encoder_outputs2[0]

        document_output_embeddings = sequence_output2.view(-1, self.document_config.hidden_size)
        masked_output_embeddings = document_output_embeddings[mask_indices, :]

        total_loss = None
        if labels is not None and label_docs is not None:
            loss_fct = CrossEntropyLoss() # ignore index -100
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.sentence_config.vocab_size), labels.view(-1))
            masked_block_loss = masked_sentence_block_loss(masked_output_embeddings, sentence_block_labels)
            print(masked_lm_loss, masked_block_loss)
            total_loss = masked_lm_loss + masked_block_loss

        if not return_dict:
            if output_hidden_states and output_attentions:
                output = (prediction_scores, sequence_output2) + (encoder_outputs1[1], encoder_outputs2[1]) + (encoder_outputs1[2], encoder_outputs2[2])
            elif output_hidden_states:
                output = (prediction_scores, sequence_output2) + (encoder_outputs1[1], encoder_outputs2[1])
            elif output_attentions:
                output = (prediction_scores, sequence_output2) + (encoder_outputs1[2], encoder_outputs2[2])
            else:
                output = (prediction_scores, sequence_output2) 

            return ((total_loss,) + output) if total_loss is not None else output

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@add_start_docstrings(
    "The bare Smith Model transformer outputting raw hidden-states without any specific head on top.",
    SMITH_START_DOCSTRING,
)
class SmithModel(SmithPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        sentence_config, document_config = parse_config(self.config)

        self.embeddings = SmithEmbeddings(sentence_config)
        self.sentence_encoder = SmithEncoder(sentence_config)
        self.sentence_pooler = SmithPooler(sentence_config)

        self.document_encoder = SmithEncoder(document_config)
        self.document_pooler = SmithPooler(document_config)

        self.sentence_config = sentence_config
        self.document_config = document_config

        self.sentence_block_mask_vector = torch.rand(self.config.hidden_size2)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(SMITH_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="smith-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        split_idx=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.sentence_config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs1 = self.sentence_encoder(
            embedding_output, 
            extended_attention_mask, 
            head_mask=head_mask, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output1 = encoder_outputs1[0]
        pooled_output1 = self.sentence_pooler(sequence_output1)

        # resplit batch into invididual documents
        pooled_output_split = torch.split(pooled_output1, split_idx, dim=0)
	
        num_sentence_blocks = max(split_idx)

        doc_input_list = []

        # pad each document with zeros so that all documents have the same shape
        for i in range(len(split_idx)):
            num_rows_to_pad = num_sentence_blocks - split_idx[i]
            zero_padding =  torch.zeros(num_rows_to_pad, self.sentence_config.hidden_size, device=device)
            doc_input_list.append(torch.cat([pooled_output_split[i], zero_padding], dim=0))
		
        sentence_block_embeddings = torch.stack(doc_input_list)

        attention_mask = torch.ones_like(sentence_block_embeddings)
        attention_mask[sentence_block_embeddings==0] = 0
        attention_mask = torch.sum(attention_mask, dim=2)
        attention_mask[attention_mask > 0] = 1

        input_shape = sentence_block_embeddings.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask=None, num_hidden_layers=self.document_config.num_hidden_layers)

        encoder_outputs2 = self.document_encoder(
            sentence_block_embeddings, 
            extended_attention_mask, 
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output2 = encoder_outputs2[0]
        pooled_output2 = self.document_pooler(sequence_output2)


        if not return_dict:
            if output_hidden_states and output_attentions:
                output = (sequence_output2, pooled_output2) + (sequence_output1, pooled_output1) + (encoder_outputs1[1], encoder_outputs2[1]) + (encoder_outputs1[2], encoder_outputs2[2])
            elif output_hidden_states:
                output = (sequence_output2, pooled_output2) + (sequence_output1, pooled_output1) + (encoder_outputs1[1], encoder_outputs2[1])
            elif output_attentions:
                output = (sequence_output2, pooled_output2) + (sequence_output1, pooled_output1) + (encoder_outputs1[2], encoder_outputs2[2])
            else:
                output = (sequence_output2, pooled_output2) + (sequence_output1, pooled_output1)

            return output

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output2,
            pooler_output=pooled_output2,
            hidden_states=encoder_outputs1.hidden_states,
            attentions=encoder_outputs1.attentions,
        )
