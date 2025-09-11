# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Less is More: On the Feature Redundancy of Pretrained Models When Transferring to Few-shot Tasks.](http://arxiv.org/abs/2310.03843) | 预训练模型在少样本任务中的特征可以极其冗余，仅使用最重要的特征维度的1%就能够达到使用完整表示时的性能。 |

# 详细

[^1]: Less is More: 关于预训练模型在少样本任务中特征冗余性的研究

    Less is More: On the Feature Redundancy of Pretrained Models When Transferring to Few-shot Tasks. (arXiv:2310.03843v1 [cs.CV])

    [http://arxiv.org/abs/2310.03843](http://arxiv.org/abs/2310.03843)

    预训练模型在少样本任务中的特征可以极其冗余，仅使用最重要的特征维度的1%就能够达到使用完整表示时的性能。

    

    将预训练模型应用于下游任务可以通过使用目标数据进行线性探测来实现，即对从预训练模型中提取的冻结特征进行训练线性分类器。由于预训练和下游数据集之间可能存在显著差异，我们可以询问是否所有预训练特征的维度对于给定的下游任务都是有用的。我们发现，在线性探测的情况下，当下游数据稀缺或少样本时，预训练特征可能极其冗余。对于一些情况，比如5类1样本任务，只使用最重要的特征维度的1%就能够达到使用完整表示时的性能。有趣的是，大部分特征只在少样本设置下是冗余的，在样本数增加时逐渐变得有用，这表明特征冗余可能是表征少样本转移问题的关键。我们给出了理论解释

    Transferring a pretrained model to a downstream task can be as easy as conducting linear probing with target data, that is, training a linear classifier upon frozen features extracted from the pretrained model. As there may exist significant gaps between pretraining and downstream datasets, one may ask whether all dimensions of the pretrained features are useful for a given downstream task. We show that, for linear probing, the pretrained features can be extremely redundant when the downstream data is scarce, or few-shot. For some cases such as 5-way 1-shot tasks, using only 1\% of the most important feature dimensions is able to recover the performance achieved by using the full representation. Interestingly, most dimensions are redundant only under few-shot settings and gradually become useful when the number of shots increases, suggesting that feature redundancy may be the key to characterizing the "few-shot" nature of few-shot transfer problems. We give a theoretical understanding 
    

