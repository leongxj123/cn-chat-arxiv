# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [No Dimensional Sampling Coresets for Classification](https://arxiv.org/abs/2402.05280) | 本文通过敏感性抽样框架提出了无维度的核心子集用于分类问题，该子集的大小与维度无关，并适用于各种损失函数和分布输入。 |

# 详细

[^1]: 无维度抽样核心子集用于分类问题

    No Dimensional Sampling Coresets for Classification

    [https://arxiv.org/abs/2402.05280](https://arxiv.org/abs/2402.05280)

    本文通过敏感性抽样框架提出了无维度的核心子集用于分类问题，该子集的大小与维度无关，并适用于各种损失函数和分布输入。

    

    我们通过敏感性抽样框架对于分类问题的核心子集的已知内容进行了精炼和概括。这种核心子集寻求输入数据的最小可能子集，以便可以在核心子集上优化损失函数，并确保对于原始数据的逼近保证。我们的分析提供了第一个无维度核心子集，因此大小与维度无关。此外，我们的结果是通用的，适用于分布输入并且可以使用独立同分布样本，因此可以提供样本复杂度边界，并适用于各种损失函数。我们开发的关键工具是主要敏感性抽样方法的Radamacher复杂度版本，这可能是一个独立感兴趣的工具。

    We refine and generalize what is known about coresets for classification problems via the sensitivity sampling framework. Such coresets seek the smallest possible subsets of input data, so one can optimize a loss function on the coreset and ensure approximation guarantees with respect to the original data. Our analysis provides the first no dimensional coresets, so the size does not depend on the dimension. Moreover, our results are general, apply for distributional input and can use iid samples, so provide sample complexity bounds, and work for a variety of loss functions. A key tool we develop is a Radamacher complexity version of the main sensitivity sampling approach, which can be of independent interest.
    

