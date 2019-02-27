# -*- coding: utf-8 -*-

from torch import nn
import torch
import torch.nn.functional as F

from modeling import BertPreTrainedModel, BertModel
import building_blocks as bb


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


class SimBertBiMPM(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(SimBertBiMPM, self).__init__(config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.d = config.hidden_size  # bert的hidden_size，即词向量的长度
        self.l = 20  # perspective的数目
        self.bimpm_config = {  # bimpm的参数配置
            "hidden_size": 128,
        }

        # ----- 词嵌入层 -----
        self.bert = BertModel(config)

        # ----- 语境表示层 -----
        self.context_LSTM = nn.LSTM(
            input_size=self.d,
            hidden_size=self.bimpm_config["hidden_size"],
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- 匹配层 -----
        for i in range(1, 9):
            setattr(self, f'mp_w{i}',
                    nn.Parameter(torch.rand(self.l, self.bimpm_config["hidden_size"])))

        # ----- 聚集层 -----
        self.aggregation_LSTM = nn.LSTM(
            input_size=self.l * 8,
            hidden_size=self.bimpm_config["hidden_size"],
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- 全连接层 -----
        self.pred_fc1 = nn.Linear(self.bimpm_config["hidden_size"] * 4, self.bimpm_config["hidden_size"] * 2)
        self.pred_fc2 = nn.Linear(self.bimpm_config["hidden_size"] * 2, self.num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids_a, input_ids_b,
                token_type_ids_a=None, token_type_ids_b=None,
                attention_mask_a=None, attention_mask_b=None,
                labels=None):

        # ----- Matching Layer -----
        def mp_matching_func(v1, v2, w):
            """
            :param v1: (batch, seq_len, hidden_size)
            :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l)
            """
            seq_len = v1.size(1)

            # Trick for large memory requirement
            """
            if len(v2.size()) == 2:
                v2 = torch.stack([v2] * seq_len, dim=1)
            m = []
            for i in range(self.l):
                # v1: (batch, seq_len, hidden_size)
                # v2: (batch, seq_len, hidden_size)
                # w: (1, 1, hidden_size)
                # -> (batch, seq_len)
                m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))
            # list of (batch, seq_len) -> (batch, seq_len, l)
            m = torch.stack(m, dim=2)
            """

            # (1, 1, hidden_size, l)
            w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
            # (batch, seq_len, hidden_size, l)
            v1 = w * torch.stack([v1] * self.l, dim=3)
            if len(v2.size()) == 3:
                v2 = w * torch.stack([v2] * self.l, dim=3)
            else:
                v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)

            m = F.cosine_similarity(v1, v2, dim=2)

            return m

        def mp_matching_func_pairwise(v1, v2, w):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l, seq_len1, seq_len2)
            """

            # Trick for large memory requirement
            """
            m = []
            for i in range(self.l):
                # (1, 1, hidden_size)
                w_i = w[i].view(1, 1, -1)
                # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
                v1, v2 = w_i * v1, w_i * v2
                # (batch, seq_len, hidden_size->1)
                v1_norm = v1.norm(p=2, dim=2, keepdim=True)
                v2_norm = v2.norm(p=2, dim=2, keepdim=True)
                # (batch, seq_len1, seq_len2)
                n = torch.matmul(v1, v2.permute(0, 2, 1))
                d = v1_norm * v2_norm.permute(0, 2, 1)
                m.append(div_with_small_value(n, d))
            # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, l)
            m = torch.stack(m, dim=3)
            """

            # (1, l, 1, hidden_size)
            w = w.unsqueeze(0).unsqueeze(2)
            # (batch, l, seq_len, hidden_size)
            v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
            # (batch, l, seq_len, hidden_size->1)
            v1_norm = v1.norm(p=2, dim=3, keepdim=True)
            v2_norm = v2.norm(p=2, dim=3, keepdim=True)

            # (batch, l, seq_len1, seq_len2)
            n = torch.matmul(v1, v2.transpose(2, 3))
            d = v1_norm * v2_norm.transpose(2, 3)

            # (batch, seq_len1, seq_len2, l)
            m = div_with_small_value(n, d).permute(0, 2, 3, 1)

            return m

        def attention(v1, v2):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :return: (batch, seq_len1, seq_len2)
            """

            # (batch, seq_len1, 1)
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            # (batch, 1, seq_len2)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

            # (batch, seq_len1, seq_len2)
            a = torch.bmm(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm

            return div_with_small_value(a, d)

        def div_with_small_value(n, d, eps=1e-8):
            # too small values are replaced by 1e-8 to prevent it from exploding.
            d = d * (d > eps).float() + eps * (d <= eps).float()
            return n / d

        # ----- 词嵌入层 -----
        word_emb_a, _ = self.bert(input_ids_a, token_type_ids_a, attention_mask_a, output_all_encoded_layers=False)
        word_emb_b, _ = self.bert(input_ids_b, token_type_ids_b, attention_mask_b, output_all_encoded_layers=False)
        word_emb_a = self.dropout(word_emb_a)
        word_emb_b = self.dropout(word_emb_b)

        # ----- 语境表示层 -----
        # (batch, seq_len, hidden_size * 2)
        con_a, _ = self.context_LSTM(word_emb_a)
        con_b, _ = self.context_LSTM(word_emb_b)
        con_a = self.dropout(con_a)
        con_b = self.dropout(con_b)
        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_a, self.bimpm_config["hidden_size"], dim=-1)
        con_h_fw, con_h_bw = torch.split(con_b, self.bimpm_config["hidden_size"], dim=-1)

        # ----- 匹配层 -----
        # 1. Full-Matching
        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching
        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)
        # (batch, seq_len, l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        # 3. Attentive-Matching
        # (batch, seq_len1, seq_len2)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)
        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))
        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        # (batch, seq_len, l)
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)

        # 4. Max-Attentive-Matching
        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)
        # (batch, seq_len, l)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)
        # (batch, seq_len, l * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)
        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.bimpm_config["hidden_size"] * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.bimpm_config["hidden_size"] * 2)], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = F.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        logits = self.pred_fc2(x)

        return logits


class SimBertABCNN1(BertPreTrainedModel):
    '''
    ABCNN1
    1. ABCNN1
    2. wide convolution
    3. W-ap

    Attributes
    ----------
    layer_size : int
        the number of (abcnn1)
    distance : function
        cosine similarity or manhattan
    abcnn : list of abcnn1
    conv : list of convolution layer
    wp : list of w-ap pooling layer
    ap : list of pooling layer
    fc : last linear layer(in paper use logistic regression)
    '''

    def __init__(self, config, num_labels, args):
        # config 是bert的config
        # args 是整个app的args
        super(SimBertABCNN1, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # ----- ABCNN1的基本参数设置 -----
        self.abcnn_config = {
            "layer_size": 2,  # 卷积-池化 层数
            "distance": bb.cosine_similarity,  # 距离衡量标准
            "inception": True,
            "filter_width": 2,
            "filter_channel": 130
        }

        # ----- abcnn 结构 -----
        self.abcnn = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList([bb.ApLayer(config.hidden_size)])
        self.wp = nn.ModuleList()
        self.fc = nn.Linear(self.abcnn_config["layer_size"] + 1, self.num_labels)

        for i in range(self.abcnn_config["layer_size"]):
            self.abcnn.append(bb.Abcnn1Portion(args.max_seq_length, config.hidden_size if i == 0 else self.abcnn_config["filter_channel"]))
            self.conv.append(
                bb.ConvLayer(False, args.max_seq_length, self.abcnn_config["filter_width"],
                             config.hidden_size if i == 0
                             else self.abcnn_config["filter_channel"], self.abcnn_config["filter_channel"],
                             self.abcnn_config["inception"]))
            self.ap.append(bb.ApLayer(self.abcnn_config["filter_channel"]))
            self.wp.append(bb.WpLayer(args.max_seq_length, self.abcnn_config["filter_width"], False))

        # ----- 初始化参数 -----
        self.apply(self.init_bert_weights)

    def forward(self, input_ids_a, input_ids_b,
                token_type_ids_a=None, token_type_ids_b=None,
                attention_mask_a=None, attention_mask_b=None,
                labels=None):
        """
        1. stack sentence vector similarity

        2. for layer_size
            abcnn1
            convolution
            stack sentence vector similarity
            W-ap for next loop x1, x2

        3. concatenate similarity list
            size (batch_size, layer_size + 1)

        4. Linear layer
            size (batch_size, 1)

        句子的词嵌入格式必须为 (batch_size, 1, sentence_length, emb_dim)

        Returns
        -------
        output : 2-D torch Tensor
            size (batch_size, 1)
        """

        # ----- bert 词嵌入 -----
        word_emb_a, _ = self.bert(input_ids_a, token_type_ids_a, attention_mask_a, output_all_encoded_layers=False)
        word_emb_b, _ = self.bert(input_ids_b, token_type_ids_b, attention_mask_b, output_all_encoded_layers=False)
        x1 = torch.unsqueeze(word_emb_a, 1)
        x2 = torch.unsqueeze(word_emb_b, 1)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        # ----- abcnn -----
        sim = []
        sim.append(self.distance(self.ap[0](x1), self.ap[0](x2)))

        for i in range(self.layer_size):
            x1, x2 = self.abcnn[i](x1, x2)
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.distance(self.ap[i + 1](x1), self.ap[i + 1](x2)))
            x1 = self.wp[i](x1)
            x2 = self.wp[i](x2)

        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output
