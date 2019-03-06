
# 上层模型开发标准
## 文本分类任务
forward
- 输入：bert_encoded_layers，size=[batch_size, seq_len, 768]
- 输出：
    - pred：size=[batch_size, 1]
    - loss：必须可直接在train循环中做backward