# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Performative Prediction with Neural Networks.](http://arxiv.org/abs/2304.06879) | 本文提出了执行预测的框架，通过找到具有执行稳定性的分类器来适用于数据分布。通过假设数据分布相对于模型的预测值可Lipschitz连续，使得我们能够放宽对损失函数的假设要求。 |

# 详细

[^1]: 神经网络下的执行预测

    Performative Prediction with Neural Networks. (arXiv:2304.06879v1 [cs.LG])

    [http://arxiv.org/abs/2304.06879](http://arxiv.org/abs/2304.06879)

    本文提出了执行预测的框架，通过找到具有执行稳定性的分类器来适用于数据分布。通过假设数据分布相对于模型的预测值可Lipschitz连续，使得我们能够放宽对损失函数的假设要求。

    

    执行预测是一种学习模型并影响其预测数据的框架。本文旨在找到分类器，使其具有执行稳定性，即适用于其产生的数据分布的最佳分类器。在使用重复风险最小化方法找到具有执行稳定性的分类器的标准收敛结果中，假设数据分布对于模型参数是可Lipschitz连续的。在这种情况下，损失必须对这些参数强凸和平滑；否则，该方法将在某些问题上发散。然而本文则假设数据分布是相对于模型的预测值可Lipschitz连续的，这是执行系统的更加自然的假设。结果，我们能够显著放宽对损失函数的假设要求。作为一个说明，我们介绍了一种建模真实数据分布的重采样过程，并使用其来实证执行稳定性相对于其他目标的效益。

    Performative prediction is a framework for learning models that influence the data they intend to predict. We focus on finding classifiers that are performatively stable, i.e. optimal for the data distribution they induce. Standard convergence results for finding a performatively stable classifier with the method of repeated risk minimization assume that the data distribution is Lipschitz continuous to the model's parameters. Under this assumption, the loss must be strongly convex and smooth in these parameters; otherwise, the method will diverge for some problems. In this work, we instead assume that the data distribution is Lipschitz continuous with respect to the model's predictions, a more natural assumption for performative systems. As a result, we are able to significantly relax the assumptions on the loss function. In particular, we do not need to assume convexity with respect to the model's parameters. As an illustration, we introduce a resampling procedure that models realisti
    

