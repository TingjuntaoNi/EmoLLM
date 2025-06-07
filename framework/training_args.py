import json
import argparse
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments, HfArgumentParser


@dataclass
class ModelEmotionArguments(TrainingArguments):
    # out_dir_root: str = field(
    #     default='outputs', metadata={"help": "Path to save checkpoints and results"}
    # )
    backbone: str = field(
        default='roberta-base', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    prompt_len: int = field(
        default=100, metadata={"help": "Length of soft prompt tokens."}
    )
    sentiment: str = field(
        default='surprise', metadata={"help": "Which sentiment to train on."} # "Number of layers of the cross-model projector." 这是原来代码里写的，疑似有误
    )
    max_source_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.sentiment = self.sentiment.lower()
        self.out_dir_root = self.output_dir


class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        
        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)
