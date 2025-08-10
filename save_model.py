
from typing import Callable, Dict, Union, Optional, Tuple, NamedTuple, Any, List
import logging
from pathlib import Path
import rich
import rich.syntax
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
import shutil
from omegaconf import OmegaConf, DictConfig
import torch
import lightning as L
from lightning.fabric.utilities.rank_zero import rank_zero_only
import os
from huggingface_hub import HfApi, create_repo, upload_folder

import os.path as osp
from torch import nn
import colorlog
from datetime import datetime
import jsonlines

# from forgetting_transformer.model.common import LMOutput
from transformers.modeling_outputs import ModelOutput
from forgetting_transformer.datamodule.common import DataInfo, Batch
from forgetting_transformer.checkpoint import Checkpointer
from configs.config import Config
from collections import defaultdict, OrderedDict
import numpy as np
import time
from dataclasses import dataclass, field, asdict
from torch.distributed.fsdp import FullyShardedDataParallel
import torch.utils.flop_counter
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
import json
import pprint
from forgetting_transformer.tokenizer import JSONGPT2Tokenizer
import argparse

REMOTE_CODE_REGISTRY = {
    "forgetting_transformer": {
        "src_subdir": "forgetting_transformer",
        "config_class": "ForgettingTransformerConfig",
        "model_class": "ForgettingTransformerForCausalLM",
    },
    "alibi": {
        "src_subdir": "alibi",
        "config_class": "TransformerConfig",
        "model_class": "TransformerForCausalLM",
    },
    "transformer": {
        "src_subdir": "transformer",
        "config_class": "TransformerConfig",
        "model_class": "TransformerForCausalLM",
    },
}

@dataclass
class ModelInfo:
    total_params: int
    trainable_params: int
    embedding_params: int
    flops_per_token: int  # Note this depends how we train the model
    non_embedding_params: int = field(init=False)

    def __post_init__(self):
        self.non_embedding_params = self.total_params - self.embedding_params



# @hydra.main(version_base=None, config_name="config", config_path="configs")
def save_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_load_dir", type=str, required=True)
    parser.add_argument("--hf_save_dir", type=str, required=True)
    parser.add_argument("--hf_model_name", type=str, required=True)
    parser.add_argument("--hf_config_name", type=str, required=True)
    parser.add_argument("--hf_load_step", type=int, required=False)
    args = parser.parse_args()


    assert args.hf_load_dir is not None
    assert args.hf_save_dir is not None
    assert args.hf_load_step is None, "You can remove this if you know what you are doing"

    args.hf_load_dir = osp.realpath(args.hf_load_dir)
    load_config_path = Path(args.hf_load_dir) / "config.yaml"
    config: Config = OmegaConf.load(load_config_path)

    assert Path(args.hf_load_dir).exists()
    # with fabric.init_module(empty_init=False):
    assert OmegaConf.is_missing(
        config.model.config, "vocab_size"
    ), "Vocab size should be left missing"
    data_info_path = Path(args.hf_load_dir) / "metrics" / "jsonlines" / "train_data_info.jsonl"
    with jsonlines.open(data_info_path) as reader:
        data_info: Dict = reader.read() 
    config.model.config.vocab_size = data_info['train_data_info/vocab_size']
    model: nn.Module = hydra.utils.instantiate(config.model)

    if args.hf_load_step is None:
        resume_step, checkpoint_path = Checkpointer.get_checkpoint_path(
            checkpoint_dir=Path(args.hf_load_dir) / "checkpoints",
            step=None,
        )
        print(f"step: {resume_step}")
        assert resume_step == config.train.max_tokens
    else:
        resume_step, checkpoint_path = Checkpointer.get_checkpoint_path(
            checkpoint_dir=Path(args.hf_load_dir) / "checkpoints",
            step=args.hf_load_step,
        )
        print(f"step: {resume_step}")
        assert resume_step == args.hf_load_step
    # if input("not checking step. proceed? (y/n)").strip() == 'y':
        # pass
    # else:
        # import sys; sys.exit()
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    del checkpoint

    if "SlimPajama" in config.datamodule._target_:
        tokenizer = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B")
    elif "LongCrawl" in config.datamodule._target_:
        tokenizer = JSONGPT2Tokenizer.from_pretrained("gpt2", add_bos_token=True, clean_up_tokenization_spaces=False, add_prefix_space=False)
    elif 'npy' in config.datamodule._target_:
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b')
    else:
        raise ValueError(f"Unknow data module {config.datamodule._target_}")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2", add_bos_token=False, clean_up_tokenization_spaces=False, add_prefix_space=False)
    tokenizer.model_max_length = data_info["train_data_info/batch_len"]
    config_name = args.hf_config_name
    config_class = REMOTE_CODE_REGISTRY[config_name]['config_class']
    model_class = REMOTE_CODE_REGISTRY[config_name]['model_class']
    model.config.auto_map = {
        "AutoConfig": f"configuration_{config_name}.{config_class}",
        "AutoModelForCausalLM": f"modeling_{config_name}.{model_class}",
    }
    model.config.model_type = config_name

    path = Path(args.hf_save_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)


    src_root = Path(f"src/forgetting_transformer/model/{config_name}")

    src_cfg = src_root / f"configuration_{config_name}.py"
    src_model = src_root / f"modeling_{config_name}.py"
    if not src_cfg.exists() or not src_model.exists():
        raise FileNotFoundError(f"Missing remote code: {src_cfg} or {src_model}")

    if src_cfg.exists():
        shutil.copyfile(src_cfg, path / f"configuration_{config_name}.py")
    if src_model.exists():
        shutil.copyfile(src_model, path / f"modeling_{config_name}.py")
    Path(path / "__init__.py").write_text("# for HF remote code\n")


    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved to {path}")
    model_name = args.hf_model_name
    model.push_to_hub(f"{model_name}")
    tokenizer.push_to_hub(f"{model_name}")
    api = HfApi()
    api.create_repo(f'xiulinyang/{model_name}', exist_ok=True)
    upload_folder(
        repo_id=model_name,
        folder_path=path,
        commit_message="add remote code + model files",
        allow_patterns="*",
    )
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2", add_bos_token=False, clean_up_tokenization_spaces=False, add_prefix_space=True)
    print(f'Model and tokenizer pushed to {model_name}')
    # import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    save_model()  # pylint: disable=no-value-for-parameter
