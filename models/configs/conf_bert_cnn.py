
"""
这个模型是Bert + CNN
"""


class ConfigBertCnn(object):
    def __init__(self,
                 num_labels=2,
                 seq_len=64,
                 dropout_after_bert=0.1,
                 conv_filter_height=2,
                 conv_filter_width=768,
                 conv_in_channel=1,
                 conv_out_channel=128
                 ):

        self.num_labels = num_labels
        self.seq_len = seq_len
        self.dropout_after_bert = dropout_after_bert

        self.conv_filter_height = conv_filter_height
        self.conv_filter_width = conv_filter_width
        self.conv_in_channel = conv_in_channel
        self.conv_out_channel = conv_out_channel
