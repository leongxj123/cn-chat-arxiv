# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Lessons on Datasets and Paradigms in Machine Learning for Symbolic Computation: A Case Study on CAD.](http://arxiv.org/abs/2401.13343) | 这项研究提出了机器学习在符号计算中的应用经验教训，包括在机器学习之前对数据集进行分析的重要性以及不同机器学习范式的选择。通过在柱面代数分解中的变量排序选择中的案例研究，发现了数据集中的不平衡问题，并引入了增强技术来改善数据集的平衡性。 |

# 详细

[^1]: 机器学习在符号计算中的数据集和范式的教训：基于CAD的案例研究

    Lessons on Datasets and Paradigms in Machine Learning for Symbolic Computation: A Case Study on CAD. (arXiv:2401.13343v1 [cs.SC])

    [http://arxiv.org/abs/2401.13343](http://arxiv.org/abs/2401.13343)

    这项研究提出了机器学习在符号计算中的应用经验教训，包括在机器学习之前对数据集进行分析的重要性以及不同机器学习范式的选择。通过在柱面代数分解中的变量排序选择中的案例研究，发现了数据集中的不平衡问题，并引入了增强技术来改善数据集的平衡性。

    

    符号计算算法及其在计算机代数系统中的实现通常包含一些选择，这些选择不会影响结果的正确性，但对所需资源有显著影响：利用机器学习模型可以针对每个问题单独做出这些选择。本研究报告了在符号计算中使用机器学习的经验教训，特别是在机器学习之前对数据集进行分析的重要性，以及可能使用的不同机器学习范式。下面以柱面代数分解中的变量排序选择作为一个具体案例研究的结果来展示，但认为所学到的教训适用于符号计算中的其他决策。我们利用一个现有的从应用中导出的示例数据集，发现该数据集在变量排序决策方面存在不平衡。我们引入了一个多项式系统问题的增强技术，使得每个问题可以有多个示例以增强数据集的平衡性。

    Symbolic Computation algorithms and their implementation in computer algebra systems often contain choices which do not affect the correctness of the output but can significantly impact the resources required: such choices can benefit from having them made separately for each problem via a machine learning model. This study reports lessons on such use of machine learning in symbolic computation, in particular on the importance of analysing datasets prior to machine learning and on the different machine learning paradigms that may be utilised. We present results for a particular case study, the selection of variable ordering for cylindrical algebraic decomposition, but expect that the lessons learned are applicable to other decisions in symbolic computation.  We utilise an existing dataset of examples derived from applications which was found to be imbalanced with respect to the variable ordering decision. We introduce an augmentation technique for polynomial systems problems that allow
    

