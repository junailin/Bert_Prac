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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(curr_dir, '../'))
import random
from torch.nn import CrossEntropyLoss

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from parallel import DataParallelModel, DataParallelCriterion
from utils import DataProcessor, InputExample, InputFeatures, convert_examples_to_features, _truncate_seq_pair, accuracy
from ner_modeling import BertCRF ,CrfLossVb

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class Ner208Processor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, seq_length):
        self.seq_length = seq_length

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_infer_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "infer")

    def get_labels(self):
        """See base class."""
        return ["I-PER", "O", "I-LOC", "I-ORG", "SEP"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
        return: InputExample:
            examples_fake - label是假的
            all_labels - 二维数据，保存了所有的真的labels
        """
        examples = []
        all_labels = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            labels = line[1].strip().split(" ")
            label_dict = {}  # 功能用create_features()里的label_map
            index = 0
            for label in self.get_labels():
                label_dict[label] = index
                index += 1
            for i in range(len(labels)):
                labels[i] = label_dict.get(labels[i])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels[0]))
            # label长度处理，使之等于max_seq_length
            if len(labels) < self.seq_length:
                for k in range(self.seq_length - len(labels)):
                    labels.append("SEP")
            if len(labels) > self.seq_length:
                labels = labels[:self.seq_length]
            all_labels.append(labels)
        examples_fake = examples
        return examples_fake, all_labels


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="/workspace/dataset/CoLA",  # /workspace/dataset/CoLA
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default="/workspace/train_output/test_bert_classifier",  # /workspace/pretrained_models/bert_en
                        type=str,
                        help="填bert预训练模型(或者是已经fine-tune的模型)的路径，路径下必须包括以下三个文件"
                        "pytorch_model.bin  vocab.txt  bert_config.json")
    parser.add_argument("--task_name",
                        default="cola",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="/workspace/train_output/test_bert_classifier",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=30,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_infer",
                        action='store_true',
                        help="Whether to run inference.")
    parser.add_argument("--eval_freq",
                        default=20,
                        help="训练过程中评估模型的频率，即多少个 step 评估一次模型")
    # parser.add_argument("--do_eval",
    #                     action='store_true',
    #                     help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=480,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=480,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--infer_batch_size",
                        default=480,
                        type=int,
                        help="Total batch size for infer.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

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

    processor = Ner208Processor(args.max_seq_length)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples, train_labels = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
    model = BertCRF.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
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
        # model = torch.nn.DataParallel(model)
        model = DataParallelModel(model)

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
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        eval_examples, eval_labels = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f for f in train_labels], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                predictions = model(input_ids, segment_ids, input_mask, label_ids)  # num_gpu个 每个的shape:
                # [batch_size/?, seq_len, tag_size, tag_size]
                # for i in range(len(predictions)):
                #     predictions[i] = predictions[i].view(-1, num_labels)
                predictions = torch.cat(predictions)  # 把多个GPU的预测结果整合在一起
                predictions = predictions.view(args.max_seq_length, args.train_batch_size, num_labels, num_labels)
                loss_fct = CrfLossVb(num_labels, start_tag=101, end_tag=102, average_batch=True)
                loss_fct_parallel = DataParallelCriterion(loss_fct)
                label_ids = label_ids.view(args.max_seq_length, args.train_batch_size)
                label_ids = torch.unsqueeze(label_ids, 2)
                input_mask = input_mask.view(args.max_seq_length, args.train_batch_size)
                loss = loss_fct_parallel(predictions, label_ids, input_mask)
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
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f for f in eval_labels], dtype=torch.long)
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            eval_preds = model(input_ids, segment_ids, input_mask, label_ids)

                        # 计算loss
                        predictions = torch.cat(predictions)  # 把多个GPU的预测结果整合在一起
                        predictions = predictions.view(args.max_seq_length, args.train_batch_size, num_labels,
                                                       num_labels)
                        label_ids = label_ids.view(args.max_seq_length, args.train_batch_size)
                        label_ids = torch.unsqueeze(label_ids, 2)
                        input_mask = input_mask.view(args.max_seq_length, args.train_batch_size)
                        loss = loss_fct_parallel(predictions, label_ids, input_mask)
                        if n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        tmp_eval_loss = loss

                        eval_preds = torch.cat(eval_preds)  # shape: [batch_size, num_labels]
                        logits = eval_preds.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        tmp_eval_accuracy = accuracy(logits, label_ids)

                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'loss': loss}
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    # else:
    #     model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    # model.to(device)

    if args.do_infer:  # todo 如果 decode crf
        infer_examples, infer_labels = processor.get_infer_examples(args.data_dir)
        infer_features = convert_examples_to_features(
            infer_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running Inference *****")
        logger.info("  Num examples = %d", len(infer_examples))
        logger.info("  Batch size = %d", args.infer_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in infer_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in infer_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in infer_features], dtype=torch.long)
        all_label_ids = torch.tensor([f for f in infer_labels], dtype=torch.long)
        infer_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        infer_sampler = SequentialSampler(infer_data)
        infer_dataloader = DataLoader(infer_data, sampler=infer_sampler, batch_size=args.infer_batch_size)

        model.eval()

        for input_ids, input_mask, segment_ids, label_ids in tqdm(infer_dataloader, desc="Inference"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                infer_preds = model(input_ids, segment_ids, input_mask, label_ids)

            for i in range(len(infer_preds)):
                infer_preds[i] = infer_preds[i].view(-1, num_labels)

            infer_preds = torch.cat(infer_preds)  # shape: [batch_size, num_labels]
            logits = infer_preds.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)
            print(outputs)
        logger.info("***** Infer finished *****")


if __name__ == "__main__":
    main()