from __future__ import print_function, absolute_import
import torch
import torch.nn as nn

from building_blocks.blocks.bert import BertPreTrainedModel, BertModel


class BertCnn(BertPreTrainedModel):
    def __init__(self, config, num_labels, seq_len):
        super(BertCnn, self).__init__(config)
        # todo 参数化
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(0.1)

        # ----- bert emb 层 -----
        self.bert = BertModel(config)
        # for p in self.parameters():  # 固定 bert 的参数
        #     p.requires_grad = False

        # ----- 卷积层 -----
        conv_config = {
            "filter_height": 2,
            "filter_channel": 32
        }
        self.conv = convolution(in_channel=1,
                                word_emb_size=config.hidden_size,
                                filter_height=conv_config["filter_height"],
                                filter_channel=conv_config["filter_channel"])

        # ----- 池化层 -----
        self.ap = nn.AvgPool2d((self.seq_len-conv_config["filter_height"]+1, 1))

        # ----- 全连接层 -----
        self.classifier = nn.Linear(conv_config["filter_channel"], num_labels)

        self.apply(self.init_bert_weights)

        print("----- 各个参数的训练情况如下 -----")
        for d in self.named_parameters():
            print(d[0], ":", d[1].requires_grad)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # ----- bert emb 层 -----
        seq_emb, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        seq_emb = self.dropout(seq_emb)  # shape: [batch_size, seq_len, emb_size(768)]
        seq_emb = torch.unsqueeze(seq_emb, 1)  # shape: [batch_size, 1, seq_len, emb_size]

        # ----- 卷积层 -----
        seq_conv_out = self.conv(seq_emb)  # shape: [batch_size, filter_channel, seq_len-filter_height+1, 1]

        # ----- 池化层 -----
        seq_ap_out = self.ap(seq_conv_out)  # shape: [batch_size, filter_channel, 1, 1]

        seq_ap_out = torch.squeeze(seq_ap_out)  # shape: [batch_size, filter_channel]

        # ----- 全连接层 -----
        logits = self.classifier(seq_ap_out)  # shape: [batch_size, num_labels]

        return logits


def convolution(in_channel, word_emb_size, filter_height, filter_channel):
    """
    卷积层
    :param in_channel: 通常是 1
    :param word_emb_size: 就是句子长度，也即卷积层宽度
    :param filter_height: 一次考虑多少个单词
    :param filter_channel: 卷积之后的channel个数
    :return: 卷积层模型

    模型输入shape：[batch_size, channel_in, seq_len, emb_size]
    模型输出shape：[batch_size, filter_channel, seq_len-filter_height+1, 1]
    """
    filter_width = word_emb_size
    model = nn.Sequential(
        nn.Conv2d(in_channel, filter_channel, (filter_height, filter_width), stride=1),
        nn.BatchNorm2d(filter_channel),
        nn.Tanh()
    )
    return model
