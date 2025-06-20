import sys
import os
# from transformers import tokenizer
from transformers import AutoTokenizer
from framework.roberta_origin import RobertaForMaskedLMPrompt
from framework.training_args import ModelEmotionArguments, RemainArgHfArgumentParser
from framework.trainer import ModelEmotionTrainer
from framework.glue_metrics import simple_accuracy, acc_and_f1
from transformers import AutoConfig

# 定义计算指标函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return acc_and_f1(preds, labels)

# 解析命令行参数或 JSON 文件
parser = RemainArgHfArgumentParser(ModelEmotionArguments)
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    json_file=os.path.abspath(sys.argv[1])
    args = parser.parse_json_file(json_file, return_remaining_args=True)[0] #args = arg_string, return_remaining_strings=True) #parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    args = parser.parse_args_into_dataclasses()[0]

# Model and tokenizer initialization
# tokenizer = AutoTokenizer.from_pretrained(
#     args.backbone,
#     #max_length=args.max_source_length,
#     max_length=args.max_source_length + args.prompt_len + 1,
#     use_fast=False
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     args.backbone,
#     use_fast=False
# )
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(
    "model/roberta-base",
    use_fast=False # False是使用python版本的分词器，速度较慢但是兼容性较好；True是使用Rust实现的快速分词器。
)

# config = AutoConfig.from_pretrained(args.backbone)
# config.mask_token_id = tokenizer.mask_token_id

# model = RobertaForMaskedLMPrompt.from_pretrained(
#     args.backbone,
#     prompt_len=args.prompt_len,
#     num_labels=2
# )

# 模型初始化
model = RobertaForMaskedLMPrompt.from_pretrained(
    "model/roberta-base",
    prompt_len=args.prompt_len,
    num_labels=2
)

# Trainer initialization 初始化训练器
trainer = ModelEmotionTrainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
train_result = trainer.train_prompt()

# Evaluate the model
eval_result = trainer.eval_prompt()

