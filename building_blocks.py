import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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