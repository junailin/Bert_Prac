！整个rep处在开发阶段，很多功能未封装完毕（部分代码需手动修改）。

# 等价性模型
- 模型 forward 的结果一定是logits，loss一律在外部算，否则不好做负载均衡。

# Usage
- 模型的train、infer代码都写在`run_xxx.py`文件里
- 针对不同的数据集需要写对应的`XxxProcessor`，具体写法可参考范例
- 参数的修改在`run_config()`函数里，也可以在命令行中直接设置参数
- `main()`中的代码，除了processor之外，都不用修改

# 一些问题
重要：
- ner 任务中，label的排序方式不同，会导致模型的效果不同。比如把O放在第一位和放在后面几位，效果差的非常大。

不重要：
- siamese_bert_abcnn.py的代码结构要调整，abcnn的部分不应该写在该文件中
