# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-03-05
Description : run文件的参数类
by : wcy
"""
# import modules


# define class
class RunConfig(object):
    def __init__(self,
                 bert_model_dir="/workspace/pretrained_models/bert_en",
                 upper_model_path=None,
                 output_dir="/workspace/train_output/cola_test",
                 max_seq_length=512,
                 eval_freq=20,
                 train_batch_size=480,
                 eval_batch_size=480,
                 infer_batch_size=480,
                 learning_rate=2e-5,
                 num_train_epochs=5,
                 label_list=[0, 1],
                 fix_bert=False,
                 fix_upper=False,
                 # ----- 非重要参数 -----
                 do_lower_case=True,
                 warmup_proportion=0.1,
                 random_seed=42,
                 gradient_accumulation_steps=1,
                 fp16=False, loss_scale=0, server_ip="", server_port=""):
        """
        run 文件的运行参数
        :param upper_model_path: 上层模型的预训练模型路径（注意不是dir）
        :param bert_model_dir: 填bert预训练模型(或者是已经fine-tune的模型)的目录，路径下必须包括以下三个文件：
                                                    pytorch_model.bin / vocab.txt / bert_config.json
        :param max_seq_length: 最大句子长度，如果当前模型的下层输入模型依赖于Bert 的话，那么此时的
                                    max_seq_length 为句子最大长度 + 2
        :param label_list: 标记的标签的集合，如果当前模型的下层输入为Bert 并且模型的为序列预测问题，那么此时labe_list
                                    中必须要包含X变迁，这个标签会在Bert对输入文本进行处理的时候用到。
        :param fix_bert: 固定bert使其不更新参数
        :param fix_upper: 固定上层模型，使其不更新参数。这个参数不直接传给app类，而是传给上层模型的类
        :param output_dir: 训练好的模型的保存地址
        :param eval_freq:  训练过程中评估模型的频率，即多少个 iteration 评估一次模型
        :param train_batch_size: Total batch size for training
        :param eval_batch_size: Total batch size for eval.
        :param infer_batch_size: Total batch size for infer.
        :param do_lower_case: if lower case
        :param learning_rate: 学习率
        :param num_train_epochs: 总共训练多少个epochs
        :param warmup_proportion: Proportion of training to perform linear learning rate warmup for.
                                        E.g., 0.1 = 10%% of training.
        :param random_seed: 随机数种子
        :param gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update.
        :param fp16  后面这三个参数只有在涉及tensorRT 时才会涉及到
        :param loss_scale
        :param server_ip
        :param server_port
        """
        self.upper_model_path = upper_model_path
        self.bert_model_dir = bert_model_dir
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.infer_batch_size = infer_batch_size
        self.do_lower_case = do_lower_case
        self.learning_rate = learning_rate
        self.fix_bert = fix_bert
        self.fix_upper = fix_upper
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.random_seed = random_seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
