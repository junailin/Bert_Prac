# 等价性模型
- 模型 forward 的结果一定是logits，loss一律在外部算，否则不好做负载均衡。