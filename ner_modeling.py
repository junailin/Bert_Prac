from torch import nn
import torch

from modeling import BertPreTrainedModel, BertModel


class CRF(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Ma et al. 2016, has more parameters than CRF_S
    args:
        hidden_dim : input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans
    """

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size, bias=if_bias)

    def forward(self, feats):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        return self.hidden2tag(feats).view(-1, self.tagset_size, self.tagset_size)


class BertCRF(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertCRF, self).__init__(config)
        self.num_labels = num_labels
        self.tagset_size = config.tagset_size
        self.bert = BertModel(config)
        self.crf = CRF(config.hidden_size, self.tagset_size, True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        word_emb, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        word_emb = self.dropout(word_emb)
        crf_out = self.crf(word_emb)
        return crf_out  # shape: [batch_size, seq_len, tag_size, tag_size]


class CrfLossVb(nn.Module):
    """loss for viterbi decode
    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )
    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CrfLossVb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """

        # calculate batch size and seq len
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        # calculate sentence score
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        # calculate forward partition score

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :].clone()  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = log_sum_exp(cur_values, self.tagset_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))  #0 for partition, 1 for cur_partition

        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        # average = mask.sum()

        # average_batch
        if self.average_batch:
            loss = (partition - tg_energy) / bat_size
        else:
            loss = (partition - tg_energy)

        return loss


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M
