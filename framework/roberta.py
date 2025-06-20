import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaPreTrainedModel, 
    RobertaModel, 
    RobertaEmbeddings, 
    create_position_ids_from_input_ids,
    RobertaLMHead
)
from transformers.activations import ACT2FN, gelu
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaPooler


# 将 Prompt 嵌入整合到 RoBERTa 模型中，并实现基于 Masked Language Modeling (MLM) 的逻辑，同时支持情感分类任务的定制逻辑
class RobertaForMaskedLMPrompt(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, prompt_len=100, init_ids=None):
        super().__init__(config)

        self.prompt_len = prompt_len # Prompt 的长度，表示提示 Token 的数量

        self.roberta = RobertaModelWarp(config, prompt_len, init_ids, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config) # 用来预测 [MASK] 的内容

        self.mask_token_id = getattr(config, "mask_token_id", 50264) # 默认值为 50264，它是 [MASK] 的索引

        # The LM head weights require special treatment only when they are tied with the word embeddings
        #self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

        self.freeze_lm() # 只允许 Prompt 嵌入部分的权重更新，避免优化 RoBERTa 主体部分

    # 冻结权重
    def freeze_lm(self):
        for name, param in self.named_parameters():
            if not 'prompt_embedding' in name:
                param.requires_grad = False

    # 嵌入操作
    ## 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    ## 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    
    # 前向传播
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_id=50264,
    ):

        # 处理 Attention Mask
        if attention_mask is not None:
            device = attention_mask.device
        elif input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device
        else:
            device = torch.device("cpu")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
        
        # 嵌入拼接
        ## 原始 embedding
        text_embeds = self.roberta.embeddings.word_embeddings(input_ids)
        prompt_embeds = self.roberta.embeddings.prompt_embeddings(
            torch.arange(self.prompt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        mask_embeds = self.roberta.embeddings.word_embeddings(
            torch.full((batch_size, 1), self.mask_token_id, dtype=torch.long, device=device)
        )

        
        inputs_embeds = torch.cat([
            text_embeds[:, :1, :],  # 原始文本 "[CLS]"
            prompt_embeds, # Prompt 嵌入
            mask_embeds, # Mask Token 嵌入
            text_embeds[:, 1:, :]  # 原始文本的其他部分
        ], dim=1)

        ## 注意力掩码拼接
        ### 添加与 Prompt 和 [MASK] 对应的注意力掩码
        prompt_mask = torch.ones(batch_size, self.prompt_len, device=device)
        mask_mask   = torch.ones(batch_size, 1, device=device)
        attention_mask = torch.cat([prompt_mask, mask_mask, attention_mask], dim=1)

        # print("text_embeds:", text_embeds.shape)        # 应该是 (B, 128, 768)
        # print("inputs_embeds:", inputs_embeds.shape)    # 应该是 (B, 229, 768)
        # print("attention_mask:", attention_mask.shape)  # 应该是 (B, 229)

        # ## Add <MASK> to input_ids
        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output) # (batch_size, sequence_length, vocab_size)

        # 分类逻辑
        # "positive":22173, "negative":33407
        # "yes":10932, "no":2362
        # '1': 134, '-': 12
        # 'true': 29225, 'false': 22303
        mask_position = 1 + self.prompt_len
        mask_logits = prediction_scores[:, mask_position, :]
        logits = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:, 22173].unsqueeze(1)], dim=1)

        # 损失计算
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

# 自定义的语言模型头（Language Model Head）类
## 只保留了 dense 层和 layer_norm 层，而省去了解码层（即将隐层表示映射到词表 logits 的部分）
class RobertaLMHeadWarp(nn.Module):
    """WARP Roberta LM Head without last decoder layer."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        return x


class RobertaWarp(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, prompt_len=100, init_ids=None):
        super().__init__(config)

        self.prompt_len = prompt_len

        self.roberta = RobertaModelWarp(config, prompt_len, init_ids, add_pooling_layer=False)
        self.lm_decoder = RobertaLMHeadWarp(config)
        self.label_embedding = nn.Linear(config.hidden_size, config.num_labels)
        self.sigm = nn.Sigmoid()

        self.init_weights()

        self.freeze_lm()

    def freeze_lm(self):
        for name, param in self.named_parameters():
            if not ('prompt_embedding' in name or 'label_embedding' in name):
                param.requires_grad = False

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_id=50264,
    ):
        """
            mask_id: token ID of <mask>
        """

        if attention_mask is not None:
            device = attention_mask.device
        elif input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device
        else:
            device = torch.device("cpu")

        batch_size, seq_length = input_ids.shape

        ## Add <MASK> to input_ids
        mask_ids = torch.tensor([mask_id]).repeat(batch_size, 1).to(device)
        input_ids = torch.cat([mask_ids, input_ids], dim=1)
        
        # ## Add prefix to attention_mask
        # prompt_attention_mask = torch.ones(self.prompt_len + 1).repeat(batch_size, 1).to(attention_mask.device)
        # attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        prompt_attention_mask = torch.ones(batch_size, self.prompt_len, device=attention_mask.device)
        mask_attention_mask   = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([prompt_attention_mask, mask_attention_mask, attention_mask], dim=1)
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state[:, 0, :]

        lm_score = self.lm_decoder(hidden_states)
        logits = self.label_embedding(lm_score)
        # logits = self.sigm(logits)

        # print('shape', outputs.last_hidden_state.shape, mask.shape, hidden_states.shape, lm_score.shape, logits.shape, labels.shape)

        masked_lm_loss = None
        if labels is not None:
            # label_emb = self.label_embedding(labels)
            # loss_fct = nn.BCELoss()
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.num_labels), labels)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
        )
    
    def tie_weights(self):
        """No need to tie input and output embeddings."""
        pass



class RobertaEmbeddingsWarp(RobertaEmbeddings):
    def __init__(self, config, prompt_len, init_ids=None):
        super().__init__(config)
        # —— 保存 config，备份 mask_token_id ——
        self.config = config
        # 如果 config 里有 mask_token_id 就用它，否则用 50264 （Roberta 的默认 mask id）
        self.mask_token_id = getattr(config, "mask_token_id", 50264)

        self.prompt_len = prompt_len
        self.prompt_embeddings = nn.Embedding(prompt_len, config.hidden_size)
        if init_ids is not None:
            with torch.no_grad():
                init_embeds = self.word_embeddings(init_ids)
                self.prompt_embeddings.weight.copy_(init_embeds)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if inputs_embeds is not None:
            # 已经是拼好的 embedding，直接加 position + LN + dropout
            device = inputs_embeds.device
            batch_size, seq_length = inputs_embeds.shape[:2]

            if position_ids is None:
                position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

            embeddings = inputs_embeds + self.position_embeddings(position_ids)
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings


class RobertaModelWarp(RobertaModel):
    def __init__(self, config, prompt_len, init_ids=None, add_pooling_layer=True):
        super().__init__(config)
        self.embeddings = RobertaEmbeddingsWarp(config, prompt_len, init_ids)
        if add_pooling_layer:
            self.pooler = RobertaPooler(config)
        self.init_weights()
