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
import logging
import os

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

from models.bert import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from models.bert_cnn import BertCnn
from models.bert_capsule import BertCapsule
from util.tokenization import BertTokenizer
from util.optimization import BertAdam, warmup_linear
from util.parallel import DataParallelModel, DataParallelCriterion
from util.utils import DataProcessor, InputExample, convert_examples_to_features, accuracy

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

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
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def capsule_args(parser):
    parser.add_argument("--conv_out_channels", default=256, type=int)
    parser.add_argument("--conv_kernel_height", default=2, type=int)
    parser.add_argument("--conv_kernel_width", default=768, type=int)

    parser.add_argument("--num_primary_units", default=8, type=int)
    parser.add_argument("--conv_unit_out_channel", default=32, type=int)
    parser.add_argument("--conv_unit_kernel_height", default=2, type=int)
    parser.add_argument("--conv_unit_kernel_width", default=1, type=int)

    parser.add_argument("--output_unit_size", default=16, type=int)


def task_cola_args(parser):

    # ----- Required parameters -----
    parser.add_argument("--data_dir",
                        default="/workspace/dataset/CoLA", type=str,
                        help="训练数据的目录，这个和XxxProcessor是对应的")
    parser.add_argument("--bert_model",
                        default="/workspace/pretrained_models/bert_en", type=str,
                        help="填bert预训练模型(或者是已经fine-tune的模型)的路径，路径下必须包括以下三个文件："
                             "pytorch_model.bin  vocab.txt  bert_config.json")
    parser.add_argument("--output_dir",
                        default="/workspace/train_output/cola_test", type=str,
                        help="训练好的模型的保存地址")
    parser.add_argument("--upper_model",
                        default="Capsule", type=str)

    # ----- 重要 parameters -----
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--fix_bert", default=False, type=bool,
                        help="是否要固定住bert")
    parser.add_argument("--max_seq_length", default=32, type=int,
                        help="最大序列长度（piece tokenize 之后的）")
    parser.add_argument("--dropout_after_bert", default=0.1, type=int,
                        help="bert层之后的dropout")
    parser.add_argument("--eval_freq", default=20,
                        help="训练过程中评估模型的频率，即多少个 iteration 评估一次模型")
    parser.add_argument("--train_batch_size", default=320, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=480, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--infer_batch_size", default=480, type=int,
                        help="Total batch size for infer.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
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


def main():
    # ----- 准备参数 -----
    parser = argparse.ArgumentParser()
    task_cola_args(parser)
    capsule_args(parser)
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

    processor = ColaProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    if args.upper_model == "Linear":
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    elif args.upper_model == "CNN":
        model = BertCnn.from_pretrained(args.bert_model,
                                        num_labels=num_labels,
                                        seq_len=args.max_seq_length)
    elif args.upper_model == "Capsule":
        model = BertCapsule.from_pretrained(args.bert_model, capsule_config=args, task_config=args)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)

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
        # model = DataParallelModel(model)  # todo 如何加入负载均衡

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
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
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

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                out_digits, seq_emb = model(input_ids, segment_ids, input_mask, label_ids)
                loss = model.loss(seq_emb, out_digits, label_ids)
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
                    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
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
                            out_digits, seq_emb = model(input_ids, segment_ids, input_mask, label_ids)

                        # 计算loss
                        loss = model.loss(seq_emb, out_digits, label_ids)

                        if n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        tmp_eval_loss = loss
                        eval_loss += tmp_eval_loss.mean().item()

                        # todo 准确率测试

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    eval_loss = eval_loss / nb_eval_steps
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'eval_loss': eval_loss,
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

    if args.do_infer:
        infer_examples = processor.get_infer_examples(args.data_dir)
        infer_features = convert_examples_to_features(
            infer_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running Inference *****")
        logger.info("  Num examples = %d", len(infer_examples))
        logger.info("  Batch size = %d", args.infer_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in infer_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in infer_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in infer_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in infer_features], dtype=torch.long)
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
