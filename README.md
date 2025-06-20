# EmoLLM

该项目实现了基于 **RoBERTa** 的情绪识别实验，核心思想是通过可训练的 *Soft Prompt* 来强化模型的情绪表示能力。代码主要由以下部分构成：

## 目录结构

- `framework/`：模型及训练相关代码
  - `roberta.py`、`roberta1.py`：在 RoBERTa 上加入 Soft Prompt 的改进模型
  - `trainer.py`：自定义 `Trainer`，包含提示学习训练、评估以及神经元激活分析等功能
  - `dataset.py`：加载并处理 GoEmotions 数据集
  - `training_args.py`：训练过程所需的参数定义
  - `glue_metrics.py`：用于计算准确率、F1 等评价指标
- `data/`：GoEmotions 数据集及预处理后的 CSV 文件
- `train.py`：模型训练入口脚本，可从命令行或 JSON 文件读取参数
- `run.bash`：示例脚本，依次训练多个情绪类别
- `config_surprise.json`：训练 "surprise" 情绪的参数示例
- `requirements.txt`、`requirements.sh`：依赖安装说明

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   # 或运行 ./requirements.sh
   ```
2. 运行示例脚本：
   ```bash
   bash run.bash
   ```
   或直接执行 `train.py`：
   ```bash
   python train.py --output_dir outputs/surprise --backbone roberta-base \
       --prompt_len 100 --sentiment surprise --max_source_length 128 \
       --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
       --learning_rate 5e-5 --num_train_epochs 10 --seed 42
   ```
   也可以通过 `config_surprise.json` 配置文件传入参数：
   ```bash
   python train.py config_surprise.json
   ```

## 关键代码说明
- **Soft Prompt 实现**：`roberta.py` 中的 `RobertaForMaskedLMPrompt` 类在原模型的嵌入层后拼接可训练的提示向量，然后在 `[MASK]` 位置预测情绪标签。
- **自定义 Trainer**：`ModelEmotionTrainer` 负责数据加载、训练及评估，并提供 `activated_neuron`、`mask_activated_neuron` 等方法，用于分析神经元激活情况。
- **数据集加载**：`EmotionDataset` 读取 `data/goemotions/*.csv`，按情绪类别构造正负样本，返回模型所需的张量。
- **训练参数**：`training_args.py` 在 `TrainingArguments` 的基础上增加了 `prompt_len`、`sentiment` 等配置，支持从 JSON 文件解析。

## 数据与文件
项目附带处理后的 GoEmotions 数据集（`data/goemotions/`）和原始 CSV (`data/goemotions_*.csv`)。`files/` 目录中存放项目相关的论文及材料。

## 许可证
代码以 Apache 2.0 协议发布，数据集使用遵循其原作者的许可。
