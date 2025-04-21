# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Huber quantile regression networks.](http://arxiv.org/abs/2306.10306) | DHQRN可以预测更一般的Huber分位数，并且在预测分布的尾部提供更好的预测。 |

# 详细

[^1]: 深度Huber分位数回归网络

    Deep Huber quantile regression networks. (arXiv:2306.10306v1 [stat.ML])

    [http://arxiv.org/abs/2306.10306](http://arxiv.org/abs/2306.10306)

    DHQRN可以预测更一般的Huber分位数，并且在预测分布的尾部提供更好的预测。

    

    典型的机器学习回归应用旨在通过使用平方误差或绝对误差评分函数来报告预测概率分布的均值或中位数。发出更多预测概率分布的函数（分位数和期望值）的重要性已被认为是量化预测不确定性的手段。在深度学习（DL）应用程序中，通过分位数和期望值回归神经网络（QRNN和ERNN）可以实现这一点。在这里，我们介绍了深度Huber分位数回归网络（DHQRN），它将QRNN和ERNN嵌套为边缘情况。 DHQRN可以预测Huber分位数，这是更一般的函数，因为它们将分位数和期望值作为极限情况嵌套起来。主要思想是使用Huber分位数回归函数训练深度学习算法，这与Huber分位数功能一致。作为概念验证，DHQRN被应用于预测房价的真实数据集，并与其他回归技术进行比较。我们观察到，在几个误差指标中，DHQRN胜过其他技术，在预测分布的尾部提供更好的预测。

    Typical machine learning regression applications aim to report the mean or the median of the predictive probability distribution, via training with a squared or an absolute error scoring function. The importance of issuing predictions of more functionals of the predictive probability distribution (quantiles and expectiles) has been recognized as a means to quantify the uncertainty of the prediction. In deep learning (DL) applications, that is possible through quantile and expectile regression neural networks (QRNN and ERNN respectively). Here we introduce deep Huber quantile regression networks (DHQRN) that nest QRNNs and ERNNs as edge cases. DHQRN can predict Huber quantiles, which are more general functionals in the sense that they nest quantiles and expectiles as limiting cases. The main idea is to train a deep learning algorithm with the Huber quantile regression function, which is consistent for the Huber quantile functional. As a proof of concept, DHQRN are applied to predict hou
    

