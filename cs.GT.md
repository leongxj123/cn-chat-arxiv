# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Striking a Balance: An Optimal Mechanism Design for Heterogenous Differentially Private Data Acquisition for Logistic Regression.](http://arxiv.org/abs/2309.10340) | 本论文研究了在从隐私敏感卖方收集的数据上执行逻辑回归的问题，设计了一个优化测试损失、卖方隐私和支付的加权组合的最佳机制，通过结合博弈论、统计学习理论和差分隐私的思想，解决了买方的目标函数非凸的问题，并提供了当卖方数量变大时的渐近结果。 |

# 详细

[^1]: 寻找平衡：用于异构差分隐私数据采集的逻辑回归的最佳机制设计

    Striking a Balance: An Optimal Mechanism Design for Heterogenous Differentially Private Data Acquisition for Logistic Regression. (arXiv:2309.10340v1 [cs.LG])

    [http://arxiv.org/abs/2309.10340](http://arxiv.org/abs/2309.10340)

    本论文研究了在从隐私敏感卖方收集的数据上执行逻辑回归的问题，设计了一个优化测试损失、卖方隐私和支付的加权组合的最佳机制，通过结合博弈论、统计学习理论和差分隐私的思想，解决了买方的目标函数非凸的问题，并提供了当卖方数量变大时的渐近结果。

    

    我们研究了在从隐私敏感卖方收集的数据上执行逻辑回归的问题。由于数据是私有的，卖方必须通过支付来激励他们提供数据。因此，目标是设计一个机制，优化测试损失、卖方隐私和支付的加权组合，即在多个感兴趣的目标之间寻找平衡。我们通过结合博弈论、统计学习理论和差分隐私的思想来解决这个问题。买方的目标函数可能非常非凸。然而，我们证明，在问题参数的某些条件下，可以通过变量的变换将问题凸化。我们还提供了当卖方数量变大时，买方的测试误差和支付的渐近结果。最后，我们通过将这些思想应用于一个真实的医疗数据集来展示我们的想法。

    We investigate the problem of performing logistic regression on data collected from privacy-sensitive sellers. Since the data is private, sellers must be incentivized through payments to provide their data. Thus, the goal is to design a mechanism that optimizes a weighted combination of test loss, seller privacy, and payment, i.e., strikes a balance between multiple objectives of interest. We solve the problem by combining ideas from game theory, statistical learning theory, and differential privacy. The buyer's objective function can be highly non-convex. However, we show that, under certain conditions on the problem parameters, the problem can be convexified by using a change of variables. We also provide asymptotic results characterizing the buyer's test error and payments when the number of sellers becomes large. Finally, we demonstrate our ideas by applying them to a real healthcare data set.
    

