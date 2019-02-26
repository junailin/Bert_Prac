# -*- coding: utf-8 -*-

from torch import nn
import torch

from modeling import BertPreTrainedModel, BertModel


class SimJustBert(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(SimJustBert, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids_a, input_ids_b,
                token_type_ids_a=None, token_type_ids_b=None,
                attention_mask_a=None, attention_mask_b=None,
                labels=None):
        _, pooled_output_a = self.bert(input_ids_a, token_type_ids_a, attention_mask_a, output_all_encoded_layers=False)
        _, pooled_output_b = self.bert(input_ids_b, token_type_ids_b, attention_mask_b, output_all_encoded_layers=False)
        pooled_output = torch.cat([pooled_output_a, pooled_output_b], 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
