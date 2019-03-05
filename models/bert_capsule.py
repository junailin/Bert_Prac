from __future__ import print_function, absolute_import
import torch.nn as nn
import torch
from building_blocks.blocks.bert import BertPreTrainedModel, BertModel
from building_blocks.blocks.capsule import CapsuleNetwork


class BertCapsule(BertPreTrainedModel):
    def __init__(self, bert_config, capsule_config, task_config):
        super(BertCapsule, self).__init__(bert_config)

        # ***** 相关参数 *****
        self.bert_dropout = nn.Dropout(task_config.dropout_after_bert)

        # ----- bert emb 层 -----
        self.bert = BertModel(bert_config)
        if task_config.fix_bert:  # 固定 bert 的参数
            for p in self.parameters():
                p.requires_grad = False

        # ----- capsule 网络层 -----
        conv_out_h = task_config.max_seq_length - capsule_config.conv_kernel_height + 1
        conv_out_w = bert_config.hidden_size - capsule_config.conv_kernel_width + 1
        primary_unit_size = capsule_config.conv_unit_out_channel * \
                            (conv_out_h - capsule_config.conv_unit_kernel_height+1) * \
                            (conv_out_w - capsule_config.conv_unit_kernel_width+1)
        self.capsule_network = CapsuleNetwork(image_width=bert_config.hidden_size,
                                              image_height=task_config.max_seq_length,
                                              image_channels=1,

                                              conv_outputs=capsule_config.conv_out_channels,
                                              conv_kernel_height=capsule_config.conv_kernel_height,
                                              conv_kernel_width=capsule_config.conv_kernel_width,

                                              num_primary_units=capsule_config.num_primary_units,
                                              primary_unit_size=primary_unit_size,
                                              conv_unit_out_channel=capsule_config.conv_unit_out_channel,
                                              conv_unit_kernel_height=capsule_config.conv_unit_kernel_height,
                                              conv_unit_kernel_width=capsule_config.conv_unit_kernel_width,

                                              num_output_units=task_config.num_labels,
                                              output_unit_size=capsule_config.output_unit_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # ----- bert emb 层 -----
        seq_emb, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        seq_emb = torch.unsqueeze(seq_emb, 1)  # shape: [batch_size, 1, seq_len, emb_size]
        seq_emb_dp = self.bert_dropout(seq_emb)

        # ----- capsule -----
        return self.capsule_network(seq_emb_dp), seq_emb

