# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayes-Optimal Fair Classification with Linear Disparity Constraints via Pre-, In-, and Post-processing](https://arxiv.org/abs/2402.02817) | 本文提出了一种基于贝叶斯最优的公平分类方法，通过先处理、中处理和后处理来最小化分类错误，并在给定群体公平性约束的情况下进行优化。该方法引入了线性和双线性差异度量的概念，并找到了贝叶斯最优公平分类器的形式。本方法能够处理多个公平性约束和常见情况。 |

# 详细

[^1]: 基于先处理、中处理和后处理的线性差异约束下的贝叶斯最优公平分类

    Bayes-Optimal Fair Classification with Linear Disparity Constraints via Pre-, In-, and Post-processing

    [https://arxiv.org/abs/2402.02817](https://arxiv.org/abs/2402.02817)

    本文提出了一种基于贝叶斯最优的公平分类方法，通过先处理、中处理和后处理来最小化分类错误，并在给定群体公平性约束的情况下进行优化。该方法引入了线性和双线性差异度量的概念，并找到了贝叶斯最优公平分类器的形式。本方法能够处理多个公平性约束和常见情况。

    

    机器学习算法可能对受保护的群体产生不公平的影响。为解决这个问题，我们开发了基于贝叶斯最优的公平分类方法，旨在在给定群体公平性约束的情况下最小化分类错误。我们引入了线性差异度量的概念，它们是概率分类器的线性函数；以及双线性差异度量，它们在群体回归函数方面也是线性的。我们证明了几种常见的差异度量（如人口平等、机会平等和预测平等）都是双线性的。我们通过揭示与Neyman-Pearson引理的连接，找到了在单一线性差异度量下的贝叶斯最优公平分类器的形式。对于双线性差异度量，贝叶斯最优公平分类器变成了群体阈值规则。我们的方法还可以处理多个公平性约束（如平等的几率）和受保护属性常见的情况。

    Machine learning algorithms may have disparate impacts on protected groups. To address this, we develop methods for Bayes-optimal fair classification, aiming to minimize classification error subject to given group fairness constraints. We introduce the notion of \emph{linear disparity measures}, which are linear functions of a probabilistic classifier; and \emph{bilinear disparity measures}, which are also linear in the group-wise regression functions. We show that several popular disparity measures -- the deviations from demographic parity, equality of opportunity, and predictive equality -- are bilinear.   We find the form of Bayes-optimal fair classifiers under a single linear disparity measure, by uncovering a connection with the Neyman-Pearson lemma. For bilinear disparity measures, Bayes-optimal fair classifiers become group-wise thresholding rules. Our approach can also handle multiple fairness constraints (such as equalized odds), and the common scenario when the protected attr
    

