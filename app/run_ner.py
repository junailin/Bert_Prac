# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

"""BERT finetuning runner."""

import argparse
import logging
import collections
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(curr_dir, '../'))
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from building_blocks.blocks.bert import WEIGHTS_NAME, CONFIG_NAME
from util.tokenization import load_vocab, WordpieceTokenizer
from util.optimization import BertAdam, warmup_linear
from util.utils import InputFeatures
from models.bert_lstm_crf import BertLstmCrf

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class NerChProcessor(object):
    def __init__(self, seq_length, bert_dir):
        self.seq_length = seq_length
        self.vocab = load_vocab(os.path.join(bert_dir, "vocab.txt"))
        self.labels = ["B-LOC", "I-LOC", "E-LOC", "S-LOC",
                       "B-PER", "I-PER", "E-PER", "S-PER",
                       "B-T", "I-T", "E-T", "S-T",
                       "B-ORG", "I-ORG", "E-ORG", "S-ORG",
                       "O",
                       "[PAD]", "[CLS]", "[SEP]"]
        self.label_dict = {label: i for i, label in enumerate(self.labels)}
        self.label_dict_inv = {i: label for i, label in enumerate(self.labels)}

    def get_tagset_size(self):
        return len(self.labels)

    def get_train_features(self, data_dir):
        return self.get_features(data_dir, "train_200w.txt", 50000)

    def get_dev_features(self, data_dir):
        return self.get_features(data_dir, "test.txt")

    def get_test_features(self, data_dir):
        return self.get_features(data_dir, "test.txt")

    def get_features(self, data_dir, file_name, num_sents=None):
        features = []
        fr = open(os.path.join(data_dir, file_name))
        index = 0
        sent_words_list = []
        sent_tags_list = []
        sent_counter = 0
        while True:
            if index >= 1:
                line = fr.readline()
                if not line:
                    break
                line = line.strip().split(" ")
                if len(line) > 1:
                    if self.vocab.get(line[0]) is not None:
                        sent_words_list.append(line[0])
                    else:
                        sent_words_list.append("[UNK]")
                    sent_tags_list.append(line[1])
                else:
                    if len(sent_words_list) > 0:
                        sent_counter += 1
                        # 准备好了一个句子，把它变成 bert features
                        if len(sent_words_list) > self.seq_length - 2:
                            sent_words_list = sent_words_list[:(self.seq_length-2)]
                            sent_tags_list = sent_tags_list[:(self.seq_length-2)]
                        sent_words_list = ["[CLS]"] + sent_words_list + ["[SEP]"]
                        sent_tags_list = ["[CLS]"] + sent_tags_list + ["[SEP]"]
                        input_ids = []
                        for word in sent_words_list:
                            input_ids.append(self.vocab.get(word))
                        input_mask = [1] * len(input_ids)
                        padding = [0] * (self.seq_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids = [0] * self.seq_length

                        tag_ids = []
                        sent_tags_list += ["[PAD]"] * (self.seq_length - len(sent_tags_list))
                        for tag in sent_tags_list:
                            tag_ids.append(self.label_dict.get(tag))

                        sent_words_list = []
                        sent_tags_list = []
                        features.append(
                            InputFeatures(
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=tag_ids
                            )
                        )
            index += 1
            if num_sents is not None and sent_counter >= num_sents:
                break
        fr.close()
        return features


class ConllProcessor(object):
    # todo 加入 word_piece_tokenizer
    def __init__(self, seq_length, bert_dir):
        self.seq_length = seq_length
        self.vocab = load_vocab(os.path.join(bert_dir, "vocab.txt"))
        self.labels = ["B-MISC", "I-MISC",
                       "O",
                       "B-PER", "I-PER",
                       "B-ORG", "I-ORG",
                       "B-LOC", "I-LOC",
                       "[PAD]", "[CLS]", "[SEP]"]
        self.label_dict = {label: i for i, label in enumerate(self.labels)}
        self.label_dict_inv = {i: label for i, label in enumerate(self.labels)}

    def get_tagset_size(self):
        return len(self.labels)

    def get_train_features(self, data_dir):
        return self.get_features(data_dir, "train.txt")

    def get_dev_features(self, data_dir):
        return self.get_features(data_dir, "dev.txt")

    def get_test_features(self, data_dir):
        return self.get_features(data_dir, "test.txt")

    def get_features(self, data_dir, type):
        features = []
        fr = open(os.path.join(data_dir, type))
        index = 0
        sent_words_list = []
        sent_tags_list = []
        while True:
            if index >= 2:
                line = fr.readline()
                if not line:
                    break
                line = line.strip().split(" ")
                if len(line) > 1:
                    if self.vocab.get(line[0]) is not None:
                        sent_words_list.append(line[0])
                    else:
                        sent_words_list.append("[UNK]")
                    sent_tags_list.append(line[3])
                else:
                    if len(sent_words_list) > 0:
                        # 准备好了一个句子，把它变成 bert features
                        if len(sent_words_list) > self.seq_length - 2:
                            sent_words_list = sent_words_list[:(self.seq_length-2)]
                            sent_tags_list = sent_tags_list[:(self.seq_length-2)]
                        sent_words_list = ["[CLS]"] + sent_words_list + ["[SEP]"]
                        sent_tags_list = ["[CLS]"] + sent_tags_list + ["[SEP]"]
                        input_ids = []
                        for word in sent_words_list:
                            input_ids.append(self.vocab.get(word))
                        input_mask = [1] * len(input_ids)
                        padding = [0] * (self.seq_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids = [0] * self.seq_length

                        tag_ids = []
                        sent_tags_list += ["[PAD]"] * (self.seq_length - len(sent_tags_list))
                        for tag in sent_tags_list:
                            tag_ids.append(self.label_dict.get(tag))

                        sent_words_list = []
                        sent_tags_list = []
                        features.append(
                            InputFeatures(
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=tag_ids
                            )
                        )
            index += 1
        fr.close()
        return features


class BingjianFrenchProcessor(object):
    def __init__(self, seq_length, bert_dir):
        self.seq_length = seq_length
        self.vocab = load_vocab(os.path.join(bert_dir, "vocab.txt"))
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.labels = ["I-PER", "B-PER",
                       "I-LOC", "B-LOC",
                       "I-ORG", "B-ORG",
                       "I-MISC", "B-MISC", "O",
                       "[PAD]", "[CLS]", "[SEP]"]
        self.label_dict = {label: i for i, label in enumerate(self.labels)}
        self.label_dict_inv = {i: label for i, label in enumerate(self.labels)}

    def get_tagset_size(self):
        return len(self.labels)

    def get_train_features(self, data_dir):
        return self.get_features(data_dir, "train_10w.txt")

    def get_dev_features(self, data_dir):
        return self.get_features(data_dir, "dev_1w.txt")

    def get_test_features(self, data_dir):
        return self.get_features(data_dir, "dev_1w.txt")

    def get_features(self, data_dir, file_name):
        features = []
        fr = open(os.path.join(data_dir, file_name))
        index = 0
        sent_words_list = []
        sent_tags_list = []
        sent_orig = []
        num_total_words = 0
        num_unk_words = 0
        sent_counter = 0
        while True:
            if index >= 0:
                line = fr.readline()
                if not line:
                    break
                line = line.strip().split(" ")
                if len(line) > 1:
                    num_total_words += 1
                    sent_orig.append(line[0])
                    if self.vocab.get(line[0]) is not None:
                        sent_words_list.append(line[0])
                    else:
                        word_piece = self.wordpiece_tokenizer.tokenize(line[0])[0]
                        sent_words_list.append(word_piece)
                        if word_piece == "[UNK]":
                            num_unk_words += 1
                    sent_tags_list.append(line[1])
                else:
                    if len(sent_words_list) > 0:
                        # 准备好了一个句子，把它变成 bert features
                        sent_counter += 1
                        if len(sent_words_list) > self.seq_length - 2:
                            sent_words_list = sent_words_list[:(self.seq_length-2)]
                            sent_tags_list = sent_tags_list[:(self.seq_length-2)]
                        sent_words_list = ["[CLS]"] + sent_words_list + ["[SEP]"]
                        sent_tags_list = ["[CLS]"] + sent_tags_list + ["[SEP]"]
                        input_ids = []
                        for word in sent_words_list:
                            input_ids.append(self.vocab.get(word))
                        input_mask = [1] * len(input_ids)
                        padding = [0] * (self.seq_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids = [0] * self.seq_length

                        tag_ids = []
                        sent_tags_list += ["[PAD]"] * (self.seq_length - len(sent_tags_list))
                        for tag in sent_tags_list:
                            if self.label_dict.get(tag) is None:
                                print(tag)
                            tag_ids.append(self.label_dict.get(tag))
                        if tag_ids is None:
                            print("tag_ids is None!!!")
                        if sent_counter < 6:
                            print("原始句子:\t", sent_orig)
                            print("sentence:\t", sent_words_list)
                            print("tags:\t", sent_tags_list)
                            print("input_ids:\t", input_ids)
                            print("input_mask:\t", input_mask)
                            print("segment_ids:\t", segment_ids)
                            print("tag_ids:\t", tag_ids)
                            print()

                        sent_words_list = []
                        sent_tags_list = []
                        sent_orig = []
                        features.append(
                            InputFeatures(
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=tag_ids
                            )
                        )

            index += 1
        fr.close()
        print("\n", file_name, "中的 UNK 比例为", (num_unk_words/num_total_words), "\n")
        return features


class BingjianAlbProcessor(object):
    def __init__(self, seq_length, bert_dir):
        self.seq_length = seq_length
        self.vocab = load_vocab(os.path.join(bert_dir, "vocab.txt"))
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC',
                       "[PAD]", "[CLS]", "[SEP]"]
        self.label_dict = {label: i for i, label in enumerate(self.labels)}
        self.label_dict_inv = {i: label for i, label in enumerate(self.labels)}

    def get_tagset_size(self):
        return len(self.labels)

    def get_train_features(self, data_dir):
        return self.get_features(data_dir, "train_data.txt")

    def get_dev_features(self, data_dir):
        return self.get_features(data_dir, "test_data.txt")

    def get_test_features(self, data_dir):
        return self.get_features(data_dir, "test_data.txt")

    def get_features(self, data_dir, file_name):
        features = []
        fr = open(os.path.join(data_dir, file_name))
        index = 0
        sent_words_list = []
        sent_tags_list = []
        num_total_words = 0
        num_unk_words = 0
        while True:
            if index >= 0:
                line = fr.readline()
                if not line:
                    break
                line = line.strip().split(" ")
                if len(line) > 1:
                    num_total_words += 1
                    if self.vocab.get(line[0]) is not None:
                        sent_words_list.append(line[0])
                    else:
                        word_piece = self.wordpiece_tokenizer.tokenize(line[0])[0]
                        sent_words_list.append(word_piece)
                        if word_piece == "[UNK]":
                            num_unk_words += 1
                    sent_tags_list.append(line[1])
                else:
                    if len(sent_words_list) > 0:
                        # 准备好了一个句子，把它变成 bert features
                        if len(sent_words_list) > self.seq_length - 2:
                            sent_words_list = sent_words_list[:(self.seq_length-2)]
                            sent_tags_list = sent_tags_list[:(self.seq_length-2)]
                        sent_words_list = ["[CLS]"] + sent_words_list + ["[SEP]"]
                        sent_tags_list = ["[CLS]"] + sent_tags_list + ["[SEP]"]
                        input_ids = []
                        for word in sent_words_list:
                            input_ids.append(self.vocab.get(word))
                        input_mask = [1] * len(input_ids)
                        padding = [0] * (self.seq_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids = [0] * self.seq_length
                        tag_ids = []
                        sent_tags_list += ["[PAD]"] * (self.seq_length - len(sent_tags_list))
                        for tag in sent_tags_list:
                            assert self.label_dict.get(tag) is not None, [tag, "不在tagset中"]
                            tag_ids.append(self.label_dict.get(tag))
                        sent_words_list = []
                        sent_tags_list = []
                        features.append(
                            InputFeatures(
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=tag_ids
                            )
                        )
            index += 1
        fr.close()
        print("\n", file_name, "中的 UNK 比例为", (num_unk_words/num_total_words), "\n")
        return features


def run_args():
    parser = argparse.ArgumentParser()

    # ----- Required parameters -----
    parser.add_argument("--data_dir",
                        default="/workspace/dataset/bingjian/french", type=str,
                        help="训练数据的目录，这个和XxxProcessor是对应的")
    parser.add_argument("--bert_model",
                        default="/workspace/train_output/ner_bingjian_french_5epoch", type=str,
                        help="填bert预训练模型(或者是已经fine-tune的模型)的路径，路径下必须包括以下三个文件："
                             "pytorch_model.bin  vocab.txt  bert_config.json")
    parser.add_argument("--output_dir",
                        default="/workspace/train_output/ner_bingjian_french_10epoch/", type=str,
                        help="训练好的模型的保存地址")

    # ----- 重要 parameters -----
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="最大序列长度（piece tokenize 之后的）")
    parser.add_argument("--eval_freq", default=50,
                        help="训练过程中评估模型的频率，即多少个 iteration 评估一次模型")
    parser.add_argument("--train_batch_size", default=320, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=480, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--infer_batch_size", default=480, type=int,
                        help="Total batch size for infer.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=float,
                        help="Total number of training epochs to perform.")

    # ----- 其他 parameters -----
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_infer", action='store_true', help="Whether to run inference.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', default=0, type=float,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")

    return parser.parse_args()


def main():
    args = run_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_infer:
        raise ValueError("At least one of `do_train` or `do_infer` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = BingjianFrenchProcessor(args.max_seq_length, args.bert_model)

    train_features = None
    num_train_optimization_steps = None
    if args.do_train:
        train_features = processor.get_train_features(args.data_dir)
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertLstmCrf.from_pretrained(args.bert_model, tagset_size=processor.get_tagset_size())
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model = DataParallelModel(model)  # todo 多GPU的负载均衡

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    # 加载词典 (仅仅用于可视化评估)
    vocab_dict = collections.OrderedDict()
    index = 0
    with open(os.path.join(args.bert_model, "vocab.txt"), "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab_dict[index] = token
            index += 1

    if args.do_train:
        eval_features = processor.get_dev_features(args.data_dir)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        nb_tr_steps = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                input_mask = input_mask.byte()
                total_loss, scores, tag_seq = model(input_ids, segment_ids, input_mask, label_ids)  # num_gpu个 每个的shape:
                loss = total_loss

                # todo 如何加上负载均衡

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # do eval
                if global_step % args.eval_freq == 0 and global_step > 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_features))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    correct_count_all = 0
                    count_all = 0

                    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        nb_eval_steps += 1

                        with torch.no_grad():
                            input_mask = input_mask.byte()
                            total_loss, scores, tag_seq = model(input_ids, segment_ids, input_mask, label_ids)

                        eval_loss += total_loss.mean().item()
                        # todo 用tag_seq计算准确率和召回，目前是在cpu中计算，有待优化
                        tag_pred = tag_seq
                        ori_correct_mat = (torch.tensor(tag_pred).cpu()==label_ids.cpu())
                        masked_correct_mat = torch.mul(ori_correct_mat.long(), input_mask.long().cpu())
                        correct_count = masked_correct_mat.sum().tolist()

                        correct_count_all += correct_count
                        count_all += input_mask.long().sum().tolist()

                        logger.info("***** Eval examples *****")
                        show_sent_words = ""
                        show_sent_tags_stan = ""
                        show_sent_tags_pred = ""
                        for i in range(5):
                            for j in range(input_mask[i].sum().tolist()):
                                show_sent_words += vocab_dict.get(input_ids[i].tolist()[j]) + " "
                                show_sent_tags_pred += processor.label_dict_inv.get(tag_pred[i].tolist()[j]) + " "
                                show_sent_tags_stan += processor.label_dict_inv.get(label_ids[i].tolist()[j]) + " "
                            print("句子:\t", show_sent_words)
                            print("预测的tags:\t", show_sent_tags_pred)
                            print("正确的tags:\t", show_sent_tags_stan)
                            print()
                            show_sent_words = ""
                            show_sent_tags_stan = ""
                            show_sent_tags_pred = ""

                    eval_accuracy = correct_count_all / count_all
                    logger.info("***** Eval results *****")
                    logger.info("eval_accuracy = %f", eval_accuracy)
                    logger.info("train_loss = %f", tr_loss / nb_tr_steps)
                    logger.info("eval_loss = %f", eval_loss / nb_eval_steps)

                    tr_loss = 0
                    nb_tr_steps = 0

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    if args.do_infer:
        test_features = processor.get_test_features(args.data_dir)
        logger.info("***** Running inference *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.infer_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        infer_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        infer_sampler = SequentialSampler(infer_data)
        infer_dataloader = DataLoader(infer_data, sampler=infer_sampler, batch_size=args.infer_batch_size)

        model.eval()
        for input_ids, input_mask, segment_ids, label_ids in tqdm(infer_dataloader, desc="Infer"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                input_mask = input_mask.byte()
                total_loss, scores, tag_seq = model(input_ids, segment_ids, input_mask, label_ids)

            tag_pred = tag_seq
            for i in range(len(tag_pred)):
                sent_words = ""
                sent_tags = ""
                for j in range(input_mask[i].sum().tolist()):
                    sent_words += vocab_dict.get(input_ids[i].tolist()[j]) + " "
                    sent_tags += processor.label_dict_inv.get(tag_pred[i].tolist()[j]) + " "
                print(sent_words)
                print(sent_tags + "\n")

        logger.info("***** Infer finished *****")


if __name__ == "__main__":
    main()
