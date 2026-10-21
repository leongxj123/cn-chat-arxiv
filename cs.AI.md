# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers.](http://arxiv.org/abs/2310.14421) | 本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。 |
| [^2] | [Typhoon: Towards an Effective Task-Specific Masking Strategy for Pre-trained Language Models.](http://arxiv.org/abs/2303.15619) | 本文探讨了一种针对预训练语言模型的任务特定的屏蔽框架，称为Typhoon，可在GLUE基准数据集上实现卓越的下游任务性能，尤其在MRPC数据集上表现优异。 |

# 详细

[^1]: 对AI分类器的对抗鲁棒性度量的存在性，唯一性和可扩展性研究

    On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers. (arXiv:2310.14421v1 [stat.ML])

    [http://arxiv.org/abs/2310.14421](http://arxiv.org/abs/2310.14421)

    本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。

    

    本文提出并证明了针对（局部）唯一可逆分类器、广义线性模型（GLM）和熵AI（EAI）具有最小对抗路径（MAP）和最小对抗距离（MAD）的存在性、唯一性和明确的分析计算的简单可验证的数学条件。在常见的合成基准测试数据集上，针对神经网络、提升随机森林、GLM和EAI等各类AI工具进行MAP和MAD的实际计算、比较和解释，包括双卷状螺旋线及其扩展以及两个生物医学数据问题（用于健康保险理赔预测和心脏病发作致死率分类）。在生物医学应用中，展示了MAP如何在预定义的可访问控制变量子集中提供唯一的最小患者特定风险缓解干预措施。

    Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.
    
[^2]: 台风：针对预训练语言模型的有效特定任务屏蔽策略

    Typhoon: Towards an Effective Task-Specific Masking Strategy for Pre-trained Language Models. (arXiv:2303.15619v1 [cs.CL])

    [http://arxiv.org/abs/2303.15619](http://arxiv.org/abs/2303.15619)

    本文探讨了一种针对预训练语言模型的任务特定的屏蔽框架，称为Typhoon，可在GLUE基准数据集上实现卓越的下游任务性能，尤其在MRPC数据集上表现优异。

    

    通过利用图形处理单元所能提供的高度并行性，变压器架构使自然语言处理领域取得了巨大的进展。在传统的屏蔽语言模型中，使用特殊的MASK标记来提示模型从周围单词中收集情境信息以恢复原本隐藏的信息。在本文中，我们探讨了一种预训练大型语言模型的任务特定的屏蔽框架，以在GLUE基准数据集上实现卓越的下游任务性能。我们基于记号输入梯度开发了自己的屏蔽算法Typhoon，并将其与其他标准基线进行比较。我们发现，Typhoon在MRPC数据集上的性能与整体字屏蔽相当。我们的实现可以在公共Github库中找到。

    Through exploiting a high level of parallelism enabled by graphics processing units, transformer architectures have enabled tremendous strides forward in the field of natural language processing. In a traditional masked language model, special MASK tokens are used to prompt our model to gather contextual information from surrounding words to restore originally hidden information. In this paper, we explore a task-specific masking framework for pre-trained large language models that enables superior performance on particular downstream tasks on the datasets in the GLUE benchmark. We develop our own masking algorithm, Typhoon, based on token input gradients, and compare this with other standard baselines. We find that Typhoon offers performance competitive with whole-word masking on the MRPC dataset. Our implementation can be found in a public Github Repository.
    

