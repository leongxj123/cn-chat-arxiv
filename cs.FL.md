# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpretable Anomaly Detection via Discrete Optimization.](http://arxiv.org/abs/2303.14111) | 该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。 |

# 详细

[^1]: 通过离散优化实现可解释性异常检测

    Interpretable Anomaly Detection via Discrete Optimization. (arXiv:2303.14111v1 [cs.LG])

    [http://arxiv.org/abs/2303.14111](http://arxiv.org/abs/2303.14111)

    该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。

    

    异常检测在许多应用领域中都是必不可少的，例如网络安全、执法、医学和欺诈保护。然而，目前深度学习方法的决策过程往往难以理解，这通常限制了它们的实际应用性。为了克服这个限制，我们提出了一个学习框架，可以从序列数据中学习可解释性的异常检测器。具体来说，我们考虑从给定的未标记序列多重集中学习确定性有限自动机 （DFA）的任务。我们证明了这个问题是计算难题，并基于约束优化开发了两个学习算法。此外，我们为优化问题引入了新的正则化方案，以提高我们的DFA的整体可解释性。通过原型实现，我们证明我们的方法在准确性和F1分数方面表现出有望的结果。

    Anomaly detection is essential in many application domains, such as cyber security, law enforcement, medicine, and fraud protection. However, the decision-making of current deep learning approaches is notoriously hard to understand, which often limits their practical applicability. To overcome this limitation, we propose a framework for learning inherently interpretable anomaly detectors from sequential data. More specifically, we consider the task of learning a deterministic finite automaton (DFA) from a given multi-set of unlabeled sequences. We show that this problem is computationally hard and develop two learning algorithms based on constraint optimization. Moreover, we introduce novel regularization schemes for our optimization problems that improve the overall interpretability of our DFAs. Using a prototype implementation, we demonstrate that our approach shows promising results in terms of accuracy and F1 score.
    

