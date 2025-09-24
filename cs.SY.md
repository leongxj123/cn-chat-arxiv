# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup](https://arxiv.org/abs/2306.03897) | 在无监督学习设置中，提出了一种名为DANSE的基于数据驱动的非线性状态估计方法，利用数据驱动的循环神经网络捕捉模型无关过程中的潜在非线性动态。 |

# 详细

[^1]: DANSE: 无监督学习设置中模型无关过程的基于数据驱动的非线性状态估计

    DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup

    [https://arxiv.org/abs/2306.03897](https://arxiv.org/abs/2306.03897)

    在无监督学习设置中，提出了一种名为DANSE的基于数据驱动的非线性状态估计方法，利用数据驱动的循环神经网络捕捉模型无关过程中的潜在非线性动态。

    

    我们解决了在无监督学习设置中针对模型无关过程的贝叶斯状态估计和预测任务。对于模型无关过程，我们没有任何关于过程动态的先验知识。在文章中，我们提出了DANSE——一种基于数据驱动的非线性状态估计方法。DANSE提供了给定状态的线性测量的封闭形式后验概率。此外，它还提供了预测的封闭形式后验概率。DANSE中使用数据驱动的循环神经网络（RNN）来提供状态先验的参数。先验依赖于过去的测量作为输入，然后使用当前测量作为输入找到状态的封闭形式后验概率。数据驱动的RNN捕捉模型无关过程的潜在非线性动态。DANSE的训练，主要是学习RNN的参数，是使用无监督的方法进行的。

    arXiv:2306.03897v2 Announce Type: replace-cross  Abstract: We address the tasks of Bayesian state estimation and forecasting for a model-free process in an unsupervised learning setup. For a model-free process, we do not have any a-priori knowledge of the process dynamics. In the article, we propose DANSE -- a Data-driven Nonlinear State Estimation method. DANSE provides a closed-form posterior of the state of the model-free process, given linear measurements of the state. In addition, it provides a closed-form posterior for forecasting. A data-driven recurrent neural network (RNN) is used in DANSE to provide the parameters of a prior of the state. The prior depends on the past measurements as input, and then we find the closed-form posterior of the state using the current measurement as input. The data-driven RNN captures the underlying non-linear dynamics of the model-free process. The training of DANSE, mainly learning the parameters of the RNN, is executed using an unsupervised lea
    

