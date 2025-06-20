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


class RobertaForMaskedLMPrompt(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, prompt_len=100, init_ids=None):
        super().__init__(config)

        self.prompt_len = prompt_len
        self.prompt_token_id = init_ids

        self.roberta = RobertaModelWarp(config, prompt_len, init_ids, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        #self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

        self.freeze_lm()

    def freeze_lm(self):
        for name, param in self.named_parameters():
            if not 'prompt_embedding' in name:
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        
        # 1. 获取原始文本和 MASK 的 embedding
        mask_ids = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
        # text_ids 就是原始的 input_ids
        combined_ids = torch.cat([mask_ids, input_ids], dim=1)
        raw_embeds = self.roberta.embeddings.word_embeddings(combined_ids) # Shape: [B, 1 + text_len, H]

        # 2. 获取 prompt embedding
        prompt_ids = torch.arange(self.prompt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        prompt_embeds = self.roberta.embeddings.prompt_embedding(prompt_ids) # Shape: [B, prompt_len, H]

        # 3. 按照 [prompt, MASK, text] 的顺序拼接 embeddings
        # MASK 和 text 的 embedding 已经合在一起了
        inputs_embeds = torch.cat([prompt_embeds, raw_embeds], dim=1) # Shape: [B, prompt_len + 1 + text_len, H]

        # 4. 扩展 attention_mask
        prompt_mask = torch.ones(batch_size, self.prompt_len, dtype=attention_mask.dtype, device=device)
        mask_token_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
        # 拼接成 [prompt_mask, mask_token_mask, original_text_mask]
        extended_attention_mask = torch.cat([prompt_mask, mask_token_mask, attention_mask], dim=1)
        
        
        # # 只对非 None 的张量做 sanity-check
        # assert input_ids.shape[1] == attention_mask.shape[1], \
        #     f"input_ids vs attention_mask: {input_ids.shape[1]} vs {attention_mask.shape[1]}"
        # if token_type_ids is not None:
        #     assert token_type_ids.shape[1] == input_ids.shape[1], \
        #         f"token_type_ids mismatch: {token_type_ids.shape[1]} vs {input_ids.shape[1]}"
        # if position_ids is not None:
        #     assert position_ids.shape[1] == input_ids.shape[1], \
        #         f"position_ids mismatch: {position_ids.shape[1]} vs {input_ids.shape[1]}"

        # # debug 打印也要判空
        # print("Final shapes → input_ids:", input_ids.shape,
        #     "attention_mask:", attention_mask.shape,
        #     "token_type_ids:", token_type_ids.shape if token_type_ids is not None else None,
        #     "position_ids:", position_ids.shape if position_ids is not None else None)
        
        
        outputs = self.roberta(
            input_ids=None,
            attention_mask=extended_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # [batch_size, seq_len, hidden_size]
        prediction_scores = self.lm_head(sequence_output) # [batch_size, seq_len, vocab_size]

        # "positive":22173, "negative":33407
        # "yes":10932, "no":2362
        # '1': 134, '-': 12
        # 'true': 29225, 'false': 22303
        mask_position = self.prompt_len
        mask_logits = prediction_scores[:, mask_position, :] # [batch_size, vocab_size]
        
        ## 从 vocab 中抽取用于二分类的 token
        logits = torch.cat([
            mask_logits[:, 33407].unsqueeze(1),  # "negative"
            mask_logits[:, 22173].unsqueeze(1)   # "positive"
        ], dim=1)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if not return_dict:
            return (masked_lm_loss, logits) if masked_lm_loss is not None else logits

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

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

        batch_size, seq_length = input_ids.shape

        ## Add <MASK> to input_ids
        mask_ids = torch.tensor([mask_id]).repeat(batch_size, 1).to(input_ids.device)
        input_ids = torch.cat([mask_ids, input_ids], dim=1)
        
        ## Add prefix to attention_mask
        prompt_attention_mask = torch.ones(self.prompt_len + 1).repeat(batch_size, 1).to(attention_mask.device)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
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
    """
    Difference from Hugginface source code: 
        Add prompt embedding in constructor
        Update forward pass
    """

    def __init__(self, config, prompt_len, init_ids=None):
        super().__init__(config)

        ## Different from hugginface source code
        self.prompt_len = prompt_len
        self.prompt_embedding = nn.Embedding(prompt_len, config.hidden_size)

        self.init_prompt_embedding(input_ids=init_ids)

    def init_prompt_embedding(self, input_ids=None, normal_params=None):
        if input_ids is not None or normal_params is not None:
            if input_ids is not None:
                embedding = self.word_embeddings(input_ids)

            elif normal_params is not None:
                mean = self.word_embeddings.mean()
                std = self.word_embeddings.std()
                embedding = torch.normal(mean, std)
        
            self.prompt_embedding.weight = nn.Parameter(embedding)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        return super().forward(
        input_ids=input_ids, 
        token_type_ids=token_type_ids, 
        position_ids=position_ids, 
        inputs_embeds=inputs_embeds, 
        past_key_values_length=past_key_values_length
    )

class RobertaModelWarp(RobertaModel):
    """
    Difference from Huggingface source code: Use new embedding layer instead of original RobertaEmbeddings
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, prompt_len, init_ids=None, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddingsWarp(config, prompt_len, init_ids)
