# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Individualized Multi-Treatment Response Curves Estimation using RBF-net with Shared Neurons.](http://arxiv.org/abs/2401.16571) | 我们提出了一种使用共享神经元的RBF网络的非参数化治疗效应估计方法，适用于多治疗设置。该方法能够建模治疗结果的共同性，并在贝叶斯框架下实现估计和推断，通过模拟实验证明了其数值性能，应用于真实临床数据后也得到了有趣的发现。 |

# 详细

[^1]: 使用共享神经元的RBF网络估计个体化多治疗反应曲线

    Individualized Multi-Treatment Response Curves Estimation using RBF-net with Shared Neurons. (arXiv:2401.16571v1 [stat.ME])

    [http://arxiv.org/abs/2401.16571](http://arxiv.org/abs/2401.16571)

    我们提出了一种使用共享神经元的RBF网络的非参数化治疗效应估计方法，适用于多治疗设置。该方法能够建模治疗结果的共同性，并在贝叶斯框架下实现估计和推断，通过模拟实验证明了其数值性能，应用于真实临床数据后也得到了有趣的发现。

    

    异质治疗效应估计是精确医学中的一个重要问题。我们的研究兴趣在于基于一些外部协变量，确定不同治疗方式的差异效应。我们提出了一种新颖的非参数化治疗效应估计方法，适用于多治疗设置。我们对响应曲线的非参数建模依赖于带有共享隐藏神经元的径向基函数（RBF）网络。因此，我们的模型有助于建模治疗结果的共同性。我们在贝叶斯框架下开发了估计和推断方案，并通过高效的马尔科夫链蒙特卡罗算法进行实现，适当地处理了分析各个方面的不确定性。通过模拟实验，展示了该方法的数值性能。将我们提出的方法应用于MIMIC数据后，我们得到了关于不同治疗策略对ICU住院时间和12小时SOFA评分的影响的一些有趣发现。

    Heterogeneous treatment effect estimation is an important problem in precision medicine. Specific interests lie in identifying the differential effect of different treatments based on some external covariates. We propose a novel non-parametric treatment effect estimation method in a multi-treatment setting. Our non-parametric modeling of the response curves relies on radial basis function (RBF)-nets with shared hidden neurons. Our model thus facilitates modeling commonality among the treatment outcomes. The estimation and inference schemes are developed under a Bayesian framework and implemented via an efficient Markov chain Monte Carlo algorithm, appropriately accommodating uncertainty in all aspects of the analysis. The numerical performance of the method is demonstrated through simulation experiments. Applying our proposed method to MIMIC data, we obtain several interesting findings related to the impact of different treatment strategies on the length of ICU stay and 12-hour SOFA sc
    

