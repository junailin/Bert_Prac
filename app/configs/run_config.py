
"""
    模型运行config
"""


class RunConfig(object):
    def __init__(self, data_dir="/workspace/dataset/CoLA",
                 bert_model_dir="/workspace/pretrained_models/bert_en",
                 output_dir=):
        self.data_dir = data_dir
        self.bert_model_dir=bert_model_dir
        self.data_dir = output_dir








        parser = argparse.ArgumentParser()

        # ----- Required parameters -----
        parser.add_argument("--",
                            default=, type=str,
                            help="训练数据的目录，这个和XxxProcessor是对应的")
        parser.add_argument("--",
                            default=, type=str,
                            help="填bert预训练模型(或者是已经fine-tune的模型)的路径，路径下必须包括以下三个文件："
                                 "pytorch_model.bin  vocab.txt  bert_config.json")
        parser.add_argument("--",
                            default="/workspace/train_output/cola_test", type=str,
                            help="训练好的模型的保存地址")
        parser.add_argument("--upper_model",
                            default="CNN", type=str,
                            help="从这几个模型中选择："
                                 "   Linear"
                                 "   CNN")

        # ----- 重要 parameters -----
        parser.add_argument("--max_seq_length", default=64, type=int,
                            help="最大序列长度（piece tokenize 之后的）")
        parser.add_argument("--eval_freq", default=20,
                            help="训练过程中评估模型的频率，即多少个 iteration 评估一次模型")
        parser.add_argument("--train_batch_size", default=480, type=int,
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

        return parser.parse_args()