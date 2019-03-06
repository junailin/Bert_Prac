from __future__ import absolute_import, division, print_function
import logging
import sys
import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(curr_dir, '../'))
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from util.tokenization import BertTokenizer
from util.optimization import BertAdam
from util.parallel import DataParallelModel, DataParallelCriterion
from building_blocks.blocks.bert import BertModel
from util.input_utils import convert_pds_to_examples, convert_examples_to_features
from models.text_cnn import TextCnn, TextCnnConfig
from app.configs.run_config import RunConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class AppClsBasedOnBert(object):
    """ 基于 bert 的文本分类任务 """
    def __init__(self, upper_model_class, upper_model_config, run_config):
        """
        初始化应用类
        :param upper_model_class: bert的上层模型类
        :param upper_model_config: 上层模型的配置类的实例
        :param run_config: 运行配置类
        """
        # 模型运行参数
        self.run_config = run_config
        self.upper_model_config = upper_model_config
        self.run_config.train_batch_size = \
            self.run_config.train_batch_size // self.run_config.gradient_accumulation_steps

        # 创建模型输出文件夹
        if not os.path.exists(self.run_config.output_dir):
            os.makedirs(self.run_config.output_dir)
        if not os.path.exists(os.path.join(self.run_config.output_dir, "bert_tuned")):
            os.makedirs(os.path.join(self.run_config.output_dir, "bert_tuned"))

        # 训练设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(self.device, self.n_gpu))
        random.seed(self.run_config.random_seed)  # 给各个GPU分发seed
        np.random.seed(self.run_config.random_seed)
        torch.manual_seed(self.run_config.random_seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.run_config.random_seed)

        # 准备 bert 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.run_config.bert_model_dir,
                                                       do_lower_case=self.run_config.do_lower_case)

        # 实例化模型
        self.bert_model = BertModel.from_pretrained(self.run_config.bert_model_dir)
        self.upper_model = upper_model_class(upper_model_config)
        if run_config.upper_model_path is not None:  # 加载预训练模型
            state_dict = torch.load(run_config.upper_model_path)
            self.upper_model.load_state_dict(state_dict)
        self.bert_model = self.bert_model.to(self.device)
        self.upper_model = self.upper_model.to(self.device)
        self.bert_model = torch.nn.DataParallel(self.bert_model)
        self.upper_model = torch.nn.DataParallel(self.upper_model)  # todo 考虑如何加负载均衡 (主要是loss问题)

    def do_train(self, tr_sent_pds, tr_label_pds, dev_sent_pds, dev_label_pds):
        """
        模型训练
        :param tr_sent_pds: 训练句子的 pandas.Series
        :param tr_label_pds: 训练标签的 pandas.Series
        :param dev_sent_pds: 评估句子的 pandas.Series
        :param dev_label_pds: 评估标签的 pandas.Series
        :return:
        """
        # 准备训练数据和dev数据
        train_examples = convert_pds_to_examples(tr_sent_pds, tr_label_pds, set_type="train")
        train_features = convert_examples_to_features(train_examples, self.run_config.label_list,
                                                      self.run_config.max_seq_length, self.tokenizer)
        dev_examples = convert_pds_to_examples(dev_sent_pds, dev_label_pds, set_type="train")
        dev_features = convert_examples_to_features(dev_examples, self.run_config.label_list,
                                                    self.run_config.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
        dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        train_sampler = RandomSampler(train_data)
        dev_sampler = SequentialSampler(dev_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.run_config.train_batch_size)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=self.run_config.eval_batch_size)

        # 和训练相关的参数
        num_train_optimization_steps = int(len(
            train_examples) / self.run_config.train_batch_size / self.run_config.gradient_accumulation_steps) * self.run_config.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", self.run_config.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        # 准备优化器
        param_optimizer = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.run_config.learning_rate,
                             warmup=self.run_config.warmup_proportion,
                             t_total=num_train_optimization_steps)

        # 训练迭代优化
        global_step = 0
        for _ in trange(int(self.run_config.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                self.bert_model.train()
                self.upper_model.train()

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                if self.run_config.fix_bert:
                    with torch.no_grad():
                        bert_encoded_layers, _ = self.bert_model(input_ids, segment_ids, input_mask, False)
                else:
                    bert_encoded_layers, _ = self.bert_model(input_ids, segment_ids, input_mask, False)
                pred, loss = self.upper_model(bert_encoded_layers, label_ids)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.run_config.gradient_accumulation_steps > 1:
                    loss = loss / self.run_config.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1
                if (step + 1) % self.run_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # 输出训练准确率 (正式代码中删除这一段)
                # print(pred[:100])
                # print(label_ids[:100])
                # tmp_train_accuracy = (pred == label_ids).sum().tolist()
                # logger.info("===== 训练准确率 = %f", tmp_train_accuracy / len(input_ids))

                # 定时 eval
                if global_step % self.run_config.eval_freq == 0 and global_step > 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(dev_examples))
                    logger.info("  Batch size = %d", self.run_config.eval_batch_size)
                    self.bert_model.eval()
                    self.upper_model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, len(dev_examples)

                    for input_ids, input_mask, segment_ids, label_ids in tqdm(dev_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(self.device)
                        input_mask = input_mask.to(self.device)
                        segment_ids = segment_ids.to(self.device)
                        label_ids = label_ids.to(self.device)

                        with torch.no_grad():
                            bert_encoded_layers, _ = self.bert_model(input_ids, segment_ids, input_mask, False)
                            pred, loss = self.upper_model(bert_encoded_layers, label_ids)
                        if self.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if self.run_config.gradient_accumulation_steps > 1:
                            loss = loss / self.run_config.gradient_accumulation_steps
                        tmp_eval_loss = loss
                        tmp_eval_accuracy = (pred == label_ids).sum().tolist()
                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy
                        nb_eval_steps += 1

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps
                    result = {'dev_loss': eval_loss,
                              'dev_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'train_loss': loss}
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))

        # 保存训练好的模型
        # BERT: Save a trained model and the associated configuration
        model_to_save = self.bert_model.module if hasattr(self.bert_model, 'module') \
            else self.bert_model  # Only save the model it-self
        output_model_file = os.path.join(self.run_config.output_dir, "bert_tuned", "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.run_config.output_dir, "bert_tuned", "bert_config.json")
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        # UpperModel
        model_to_save = self.upper_model.module if hasattr(self.bert_model, 'module') \
            else self.bert_model  # Only save the model it-self
        output_model_file = os.path.join(self.run_config.output_dir, self.upper_model_config.name+".bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        return

    def do_infer(self, infer_sent_pds):
        """
        模型预测
        :param infer_sent_pds: 预测数据的 pandas.Series
        :return: 预测结果
        """
        # 准备训练数据和dev数据
        fake_infer_label_pds = pd.Series([0]*infer_sent_pds.shape[0])
        infer_examples = convert_pds_to_examples(infer_sent_pds, fake_infer_label_pds, set_type="infer")
        infer_features = convert_examples_to_features(infer_examples, self.run_config.label_list,
                                                      self.run_config.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in infer_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in infer_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in infer_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in infer_features], dtype=torch.long)
        infer_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        infer_sampler = SequentialSampler(infer_data)
        infer_dataloader = DataLoader(infer_data, sampler=infer_sampler, batch_size=self.run_config.infer_batch_size)

        logger.info("***** Running inference *****")
        logger.info("  Num examples = %d", len(infer_examples))
        logger.info("  Batch size = %d", self.run_config.infer_batch_size)

        self.bert_model.eval()
        self.upper_model.eval()

        for input_ids, input_mask, segment_ids, label_ids in tqdm(infer_dataloader, desc="Inference"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                bert_encoded_layers, _ = self.bert_model(input_ids, segment_ids, input_mask, False)
                pred = self.upper_model(bert_encoded_layers)

        result_writer = open(os.path.join(self.run_config.output_dir, "infer_result.txt"), "w")
        pred = pred.tolist()
        for i in range(len(pred)):
            result_writer.write(str(pred[i]) + "\n")


def task_cola_cls():
    """ 在cola数据集上做文本分类任务 """

    # 准备数据
    cola_train_df = pd.read_csv("/workspace/dataset/CoLA/train.tsv", sep="\t")
    cola_dev_df = pd.read_csv("/workspace/dataset/CoLA/dev.tsv", sep="\t")
    cola_test_df = pd.read_csv("/workspace/dataset/CoLA/test.tsv", sep="\t")
    tr_sent_pds = cola_train_df["sentence"]
    tr_label_pds = cola_train_df["label"]
    dev_sent_pds = cola_dev_df["sentence"]
    dev_label_pds = cola_dev_df["label"]
    test_sent_pds = cola_test_df["sentence"]

    # 准备 config 文件
    app_run_config = RunConfig(
        bert_model_dir="/workspace/train_output/cola_test/bert_tuned",
        upper_model_path="/workspace/train_output/cola_test/TextCnn.bin",  # "/workspace/train_output/cola_test/TextCnn.bin"
        output_dir="/workspace/train_output/cola_test2",
        max_seq_length=64,
        eval_freq=20,
        train_batch_size=320,
        eval_batch_size=480,
        infer_batch_size=480,
        learning_rate=2e-5,
        num_train_epochs=1,
        label_list=[0, 1],
        fix_bert=False,
        fix_upper=True
    )
    text_cnn_config = TextCnnConfig(
        max_seq_length=app_run_config.max_seq_length,
        no_grad=app_run_config.fix_upper
    )
    if app_run_config.fix_bert:
        print("bert被固定")
    if app_run_config.fix_upper:
        print("上层模型被固定")

    # 准备app类
    app_cls_text_cnn = AppClsBasedOnBert(TextCnn, text_cnn_config, app_run_config)

    # 执行训练
    # app_cls_text_cnn.do_train(tr_sent_pds, tr_label_pds, dev_sent_pds, dev_label_pds)

    # 执行预测 通常不会和 do_train() 一起执行
    app_cls_text_cnn.do_infer(test_sent_pds)


if __name__ == '__main__':
    task_cola_cls()

