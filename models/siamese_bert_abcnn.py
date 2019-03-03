# -*- coding: utf-8 -*-

from torch import nn
import torch
import torch.nn.functional as F

from models.bert import BertPreTrainedModel, BertModel


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

        # for p in self.parameters():  # 固定bert
        #     p.requires_grad = False

        # ----- ABCNN1的基本参数设置 -----
        self.abcnn_config = {
            "layer_size": 2,  # 卷积-池化 层数
            "distance": cosine_similarity,  # 距离衡量标准
            "inception": True,
            "filter_width": 2,
            "filter_channel": 130
        }

        # ----- abcnn 结构 -----
        self.abcnn = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList([ApLayer(config.hidden_size)])
        self.wp = nn.ModuleList()
        self.fc = nn.Linear(self.abcnn_config["layer_size"] + 1, self.num_labels)

        for i in range(self.abcnn_config["layer_size"]):
            self.abcnn.append(Abcnn1Portion(args.max_seq_length, config.hidden_size if i == 0 else self.abcnn_config["filter_channel"]))
            self.conv.append(
                ConvLayer(False, args.max_seq_length, self.abcnn_config["filter_width"],
                             config.hidden_size if i == 0
                             else self.abcnn_config["filter_channel"], self.abcnn_config["filter_channel"],
                             self.abcnn_config["inception"]))
            self.ap.append(ApLayer(self.abcnn_config["filter_channel"]))
            self.wp.append(WpLayer(args.max_seq_length, self.abcnn_config["filter_width"], False))

        print("----- 各个参数的训练情况如下 -----")
        for d in self.named_parameters():
            print(d[0], ":", d[1].requires_grad)

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
        sim.append(self.abcnn_config["distance"](self.ap[0](x1), self.ap[0](x2)))

        for i in range(self.abcnn_config["layer_size"]):
            x1, x2 = self.abcnn[i](x1, x2)
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.abcnn_config["distance"](self.ap[i + 1](x1), self.ap[i + 1](x2)))
            x1 = self.wp[i](x1)
            x2 = self.wp[i](x2)

        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output


class Abcnn1Portion(nn.Module):
    '''Part of Abcnn1
    '''

    def __init__(self, in_dim, out_dim):
        super(Abcnn1Portion, self).__init__()
        self.batchNorm = nn.BatchNorm2d(2)
        self.attention_feature_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x1, x2):
        '''
        1. compute attention matrix
            attention_m : size (batch_size, sentence_length, sentence_length)
        2. generate attention feature map(weight matrix are parameters of the model to be learned)
            x_attention : size (batch_size, 1, sentence_length, emb_dim)
        3. stack the representation feature map and attention feature map
            x : size (batch_size, 2, sentence_length, emb_dim)
        4. batch norm(not in paper)

        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, sentence_length, emb_dim)

        Returns
        -------
        (x1, x2) : list of 4-D torch Tensor
            size (batch_size, 2, sentence_length, emb_dim)
        '''
        attention_m = attention_matrix(x1, x2)

        x1_attention = self.attention_feature_layer(attention_m.permute(0, 2, 1))
        x1_attention = x1_attention.unsqueeze(1)
        x1 = torch.cat([x1, x1_attention], 1)

        x2_attention = self.attention_feature_layer(attention_m)
        x2_attention = x2_attention.unsqueeze(1)
        x2 = torch.cat([x2, x2_attention], 1)

        x1 = self.batchNorm(x1)
        x2 = self.batchNorm(x2)

        return (x1, x2)


class Abcnn2Portion(nn.Module):
    '''Part of Abcnn2
    '''

    def __init__(self, sentence_length, filter_width):
        super(Abcnn2Portion, self).__init__()
        self.wp = WpLayer(sentence_length, filter_width, True)

    def forward(self, x1, x2):
        '''
        1. compute attention matrix
            attention_m : size (batch_size, sentence_length + filter_width - 1, sentence_length + filter_width - 1)
        2. sum all attention values for a unit to derive a single attention weight for that unit
            x_a_conv : size (batch_size, sentence_length + filter_width - 1)
        3. average pooling(w-ap)

        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, sentence_length + filter_width - 1, height)

        Returns
        -------
        (x1, x2) : list of 4-D torch Tensor
            size (batch_size, 1, sentence_length, height)
        '''
        attention_m = attention_matrix(x1, x2)
        x1_a_conv = attention_m.sum(dim=1)
        x2_a_conv = attention_m.sum(dim=2)
        x1 = self.wp(x1, x1_a_conv)
        x2 = self.wp(x2, x2_a_conv)

        return (x1, x2)


class InceptionModule(nn.Module):
    '''
    inception module(not in paper)
    first layer width is filter_width(given)
    second layer width is filter_width + 4
    third layer width is sentence_length

    this helps model to be learned(when the number of layers > 8)
    '''

    def __init__(self, in_channel, sentence_length, filter_width, filter_height, filter_channel):
        super(InceptionModule, self).__init__()
        self.conv_1 = convolution(in_channel, filter_width, filter_height,
                                  int(filter_channel / 3) + filter_channel - 3 * int(filter_channel / 3),
                                  filter_width - 1)
        self.conv_2 = convolution(in_channel, filter_width + 4, filter_height, int(filter_channel / 3),
                                  filter_width + 1)
        self.conv_3 = convolution(in_channel, sentence_length, filter_height, int(filter_channel / 3),
                                  int((sentence_length + filter_width - 2) / 2))

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(x)
        out_3 = self.conv_3(x)
        output = torch.cat([out_1, out_2, out_3], dim=1)
        return output


class ConvLayer(nn.Module):
    '''
    convolution layer for abcnn

    Attributes
    ----------
    inception : bool
        whether use inception module
    '''

    def __init__(self, isAbcnn2, sentence_length, filter_width, filter_height, filter_channel, inception):
        super(ConvLayer, self).__init__()
        if inception:
            self.model = InceptionModule(1 if isAbcnn2 else 2, sentence_length, filter_width, filter_height,
                                         filter_channel)
        else:
            self.model = convolution(1 if isAbcnn2 else 2, filter_width, filter_height, filter_channel,
                                     filter_width - 1)

    def forward(self, x):
        '''
        1. convlayer
            size (batch_size, filter_channel, height, 1)
        2. transpose
            size (batch_size, 1, height, filter_channel)

        Parameters
        ----------
        x : 4-D torch Tensor
            size (batch_size, 1, height, width)

        Returns
        -------
        output : 4-D torch Tensor
            size (batch_size, 1, height, filter_channel)
        '''
        output = self.model(x)
        output = output.permute(0, 3, 2, 1)
        return output


def cosine_similarity(x1, x2):
    '''compute cosine similarity between x1 and x2

    Parameters
    ----------
    x1, x2 : 2-D torch Tensor
        size (batch_size, 1)

    Returns
    -------
    distance : 2-D torch Tensor
        similarity result of size (batch_size, 1)
    '''
    return F.cosine_similarity(x1, x2).unsqueeze(1)


def manhattan_distance(x1, x2):
    '''compute manhattan distance between x1 and x2 (not in paper)

    Parameters
    ----------
    x1, x2 : 2-D torch Tensor
        size (batch_size, 1)

    Returns
    -------
    distance : 2-D torch Tensor
        similarity result of size (batch_size, 1)
    '''
    return torch.div(torch.norm((x1 - x2), 1, 1, keepdim=True), x1.size()[1])


def convolution(in_channel, filter_width, filter_height, filter_channel, padding):
    '''convolution layer
    '''
    model = nn.Sequential(
        nn.Conv2d(in_channel, filter_channel, (filter_width, filter_height), stride=1, padding=(padding, 0)),
        nn.BatchNorm2d(filter_channel),
        nn.Tanh()
    )
    return model


def attention_matrix(x1, x2, eps=1e-6):
    '''compute attention matrix using match score

    1 / (1 + |x · y|)
    |·| is euclidean distance

    Parameters
    ----------
    x1, x2 : 4-D torch Tensor
        size (batch_size, 1, sentence_length, width)

    Returns
    -------
    output : 3-D torch Tensor
        match score result of size (batch_size, sentence_length(for x2), sentence_length(for x1))
    '''
    eps = torch.tensor(eps)
    one = torch.tensor(1.)
    euclidean = (torch.pow(x1 - x2.permute(0, 2, 1, 3), 2).sum(dim=3) + eps).sqrt()
    return (euclidean + one).reciprocal()


class ApLayer(nn.Module):
    '''column-wise averaging over all columns
    '''

    def __init__(self, width):
        super(ApLayer, self).__init__()
        self.ap = nn.AvgPool2d((1, width), stride=1)

    def forward(self, x):
        '''
        1. average pooling
            x size (batch_size, 1, sentence_length, 1)
        2. representation vector for the sentence
            output size (batch_size, sentence_length)

        Parameters
        ----------
        x : 4-D torch Tensor
            convolution output of size (batch_size, 1, sentence_length, width)

        Returns
        -------
        output : 2-D torch Tensor
            representation vector of size (batch_size, width)
        '''
        return self.ap(x).squeeze(1).squeeze(2)


class WpLayer(nn.Module):
    '''column-wise averaging over windows of w consecutive columns

    Attributes
    ----------
    attention : bool
        compute layer with attention matrix
    '''

    def __init__(self, sentence_length, filter_width, attention):
        super(WpLayer, self).__init__()
        self.attention = attention
        if attention:
            self.sentence_length = sentence_length
            self.filter_width = filter_width
        else:
            self.wp = nn.AvgPool2d((filter_width, 1), stride=1)

    def forward(self, x, attention_matrix=None):
        '''
        if attention
            reweight the convolution output with attention matrix
        else
            average pooling

        Parameters
        ----------
        x : 4-D torch Tensor
            convolution output of size (batch_size, 1, sentence_length + filter_width - 1, height)
        attention_matrix: 2-D torch Tensor
            attention matrix between (convolution output x1 and convolution output x2) of size (batch_size, sentence_length + filter_width - 1)

        Returns
        -------
        output : 4-D torch Tensor
            size (batch_size, 1, sentence_length, height)
        '''
        if self.attention:
            pools = []
            attention_matrix = attention_matrix.unsqueeze(1).unsqueeze(3)
            for i in range(self.sentence_length):
                pools.append(
                    (x[:, :, i:i + self.filter_width, :] * attention_matrix[:, :, i:i + self.filter_width, :]).sum(
                        dim=2, keepdim=True))

            return torch.cat(pools, dim=2)

        else:
            return self.wp(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Layer') == -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
