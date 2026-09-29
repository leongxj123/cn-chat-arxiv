# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Fast Graph Search Algorithm with Dynamic Optimization and Reduced Histogram for Discrimination of Binary Classification Problem.](http://arxiv.org/abs/2401.04282) | 本研究提出了一种用于二分类问题的快速图搜索算法，通过动态优化和减少直方图的方法来提高区分结果。该算法在支持向量机模型的基础上应用，显著提高了真正例并减少了假正例。 |
| [^2] | [Differentially-Private Decision Trees and Provable Robustness to Data Poisoning.](http://arxiv.org/abs/2305.15394) | 本论文提出了一种名为PrivaTree的差分隐私决策树方法，通过使用私有直方图选择分割点来在隐私保护与模型效用之间取得更好的平衡。这种方法能够接收混合的数值和类别数据，并且能够在数据篡改方面表现出可靠性。 |

# 详细

[^1]: 快速图搜索算法与动态优化和减少直方图用于二分类问题的区分 (arXiv:2401.04282v1 [cs.LG])

    A Fast Graph Search Algorithm with Dynamic Optimization and Reduced Histogram for Discrimination of Binary Classification Problem. (arXiv:2401.04282v1 [cs.LG])

    [http://arxiv.org/abs/2401.04282](http://arxiv.org/abs/2401.04282)

    本研究提出了一种用于二分类问题的快速图搜索算法，通过动态优化和减少直方图的方法来提高区分结果。该算法在支持向量机模型的基础上应用，显著提高了真正例并减少了假正例。

    

    本研究开发了一种图搜索算法，用于找到二分类问题的最优区分路径。目标函数被定义为真正例（TP）和假正例（FP）之间变异性的差异。它使用深度优先搜索（DFS）算法来寻找自顶向下的区分路径。它提出了一种动态优化过程，以在上层优化TP，然后在下层减少FP。为了加速计算速度并提高准确性，它提出了一种带有可变箱大小的减小直方图算法，而不是循环遍历所有数据点，以找到区分的特征阈值。该算法应用于支持向量机（SVM）模型上，用于预测一个人是否健康。它显著提高了SVM结果的TP并减少了FP （例如，FP减少了90%，而TP仅损失了5%）。图搜索自动生成了39个排序的区分路径。

    This study develops a graph search algorithm to find the optimal discrimination path for the binary classification problem. The objective function is defined as the difference of variations between the true positive (TP) and false positive (FP). It uses the depth first search (DFS) algorithm to find the top-down paths for discrimination. It proposes a dynamic optimization procedure to optimize TP at the upper levels and then reduce FP at the lower levels. To accelerate computing speed with improving accuracy, it proposes a reduced histogram algorithm with variable bin size instead of looping over all data points, to find the feature threshold of discrimination. The algorithm is applied on top of a Support Vector Machine (SVM) model for a binary classification problem on whether a person is fit or unfit. It significantly improves TP and reduces FP of the SVM results (e.g., reduced FP by 90% with a loss of only\ 5% TP). The graph search auto-generates 39 ranked discrimination paths withi
    
[^2]: 差分隐私决策树与对数据篡改的可靠性证明

    Differentially-Private Decision Trees and Provable Robustness to Data Poisoning. (arXiv:2305.15394v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.15394](http://arxiv.org/abs/2305.15394)

    本论文提出了一种名为PrivaTree的差分隐私决策树方法，通过使用私有直方图选择分割点来在隐私保护与模型效用之间取得更好的平衡。这种方法能够接收混合的数值和类别数据，并且能够在数据篡改方面表现出可靠性。

    

    决策树是适用于非线性学习问题的可解释模型。关于将差分隐私引入决策树学习算法的研究已经很多，差分隐私能够确保训练数据中样本的隐私性。然而，目前用于此目的的最先进算法在获得一点点隐私保护的同时牺牲了较多的模型效用。这些解决方案引入了随机决策节点，降低了决策树的准确性，或者在标记叶子节点上使用过多的隐私预算。此外，很多方法不支持连续特征或者泄露与连续特征相关的信息。我们提出了一种基于私有直方图的新方法，称为PrivaTree，它在消耗一小部分隐私预算的同时选择合适的分割点。由此产生的决策树在隐私效用权衡方面取得了显著的提升，而且能够接受混合的数值和类别数据而不泄露与数值特征相关的信息。最后，尽管给出可靠性保证一直很难，我们的方法在数据篡改方面表现出了可靠性。

    Decision trees are interpretable models that are well-suited to non-linear learning problems. Much work has been done on extending decision tree learning algorithms with differential privacy, a system that guarantees the privacy of samples within the training data. However, current state-of-the-art algorithms for this purpose sacrifice much utility for a small privacy benefit. These solutions create random decision nodes that reduce decision tree accuracy or spend an excessive share of the privacy budget on labeling leaves. Moreover, many works do not support continuous features or leak information about them. We propose a new method called PrivaTree based on private histograms that chooses good splits while consuming a small privacy budget. The resulting trees provide a significantly better privacy-utility trade-off and accept mixed numerical and categorical data without leaking information about numerical features. Finally, while it is notoriously hard to give robustness guarantees a
    

