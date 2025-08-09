# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_alibi import TransformerConfig
from .modeling_alibi import (
    TransformerForCausalLM, TransformerModel)

AutoConfig.register(TransformerConfig.model_type, TransformerConfig)
AutoModel.register(TransformerConfig, TransformerModel)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM)



__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
