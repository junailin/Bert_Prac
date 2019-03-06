import torch
import torch.nn as nn


class TextCnnConfig(object):
    def __init__(self,
                 out_channel_num=32,
                 max_seq_length=64,
                 bert_emb_size=768,
                 num_labels=2,
                 dropout_after_bert=0.1):
        self.out_channel_num = out_channel_num
        self.max_seq_length = max_seq_length
        self.bert_emb_size = bert_emb_size
        self.num_labels = num_labels
        self.dropout_after_bert = dropout_after_bert


class TextCnn(nn.Module):
    def __init__(self, config):
        super(TextCnn, self).__init__()
        self.dropout_after_bert = nn.Dropout(config.dropout_after_bert)

        # ----- 卷积层 -----
        self.conv = nn.Sequential(
            nn.Conv2d(1, config.out_channel_num,
                      (config.max_seq_length, config.bert_emb_size),
                      stride=1),
            nn.BatchNorm2d(config.out_channel_num),
            nn.Tanh()
        )

        # ----- 池化层 -----
        self.ap = nn.AvgPool2d((config.max_seq_length-config.max_seq_length+1, 1))

        # ----- 全连接层 -----
        self.classifier = nn.Linear(config.out_channel_num, config.num_labels)

    def forward(self, bert_encoded_layers):
        seq_emb = torch.unsqueeze(bert_encoded_layers, 1)  # shape: [batch_size, 1, seq_len, emb_size]
        seq_emb = self.dropout_after_bert(seq_emb)

        # ----- 卷积层 -----
        seq_conv_out = self.conv(seq_emb)  # shape: [batch_size, filter_channel, seq_len-filter_height+1, 1]

        # ----- 池化层 -----
        seq_ap_out = self.ap(seq_conv_out)  # shape: [batch_size, filter_channel, 1, 1]
        seq_ap_out = torch.squeeze(seq_ap_out)  # shape: [batch_size, filter_channel]

        # ----- 全连接层 -----
        logits = self.classifier(seq_ap_out)  # shape: [batch_size, num_labels]

        return logits

