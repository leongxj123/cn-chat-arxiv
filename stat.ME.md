# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Time-Uniform Confidence Spheres for Means of Random Vectors](https://arxiv.org/abs/2311.08168) | 该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。 |
| [^2] | [BART-SIMP: a novel framework for flexible spatial covariate modeling and prediction using Bayesian additive regression trees.](http://arxiv.org/abs/2309.13270) | BART-SIMP是一种灵活的空间协变建模和预测的新框架，通过结合高斯过程空间模型和贝叶斯加法回归树模型，可以提供可靠的不确定性估计，并成功应用于肯尼亚家庭集群样本中的人体测量响应预测。 |

# 详细

[^1]: 随机向量均值的时间均匀置信球

    Time-Uniform Confidence Spheres for Means of Random Vectors

    [https://arxiv.org/abs/2311.08168](https://arxiv.org/abs/2311.08168)

    该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。

    

    我们推导并研究了时间均匀置信球——包含随机向量均值并且跨越所有样本量具有很高概率的置信球序列（CSSs）。受Catoni和Giulini原始工作启发，我们统一并扩展了他们的分析，涵盖顺序设置并处理各种分布假设。我们的结果包括有界随机向量的经验伯恩斯坦CSS（导致新颖的经验伯恩斯坦置信区间，渐近宽度按照真实未知方差成比例缩放）、用于子-$\psi$随机向量的CSS（包括子伽马、子泊松和子指数分布）、和用于重尾随机向量（仅有两阶矩）的CSS。最后，我们提供了两个抵抗Huber噪声污染的CSS。第一个是我们经验伯恩斯坦CSS的鲁棒版本，第二个扩展了单变量序列最近的工作。

    arXiv:2311.08168v2 Announce Type: replace-cross  Abstract: We derive and study time-uniform confidence spheres -- confidence sphere sequences (CSSs) -- which contain the mean of random vectors with high probability simultaneously across all sample sizes. Inspired by the original work of Catoni and Giulini, we unify and extend their analysis to cover both the sequential setting and to handle a variety of distributional assumptions. Our results include an empirical-Bernstein CSS for bounded random vectors (resulting in a novel empirical-Bernstein confidence interval with asymptotic width scaling proportionally to the true unknown variance), CSSs for sub-$\psi$ random vectors (which includes sub-gamma, sub-Poisson, and sub-exponential), and CSSs for heavy-tailed random vectors (two moments only). Finally, we provide two CSSs that are robust to contamination by Huber noise. The first is a robust version of our empirical-Bernstein CSS, and the second extends recent work in the univariate se
    
[^2]: BART-SIMP：一种灵活的空间协变建模和预测的新框架使用贝叶斯加法回归树

    BART-SIMP: a novel framework for flexible spatial covariate modeling and prediction using Bayesian additive regression trees. (arXiv:2309.13270v1 [stat.ME])

    [http://arxiv.org/abs/2309.13270](http://arxiv.org/abs/2309.13270)

    BART-SIMP是一种灵活的空间协变建模和预测的新框架，通过结合高斯过程空间模型和贝叶斯加法回归树模型，可以提供可靠的不确定性估计，并成功应用于肯尼亚家庭集群样本中的人体测量响应预测。

    

    在空间统计学中，预测是一个经典的挑战，将空间协变量纳入具有潜在空间效应的模型中可以极大地提高预测性能。我们希望开发出灵活的回归模型，允许在协变量结构中存在非线性和交互作用。机器学习模型已经在空间环境中提出，允许残差中存在空间依赖性，但无法提供可靠的不确定性估计。在本文中，我们研究了高斯过程空间模型和贝叶斯加法回归树（BART）模型的新组合。通过将马尔可夫链蒙特卡洛（MCMC）与嵌套拉普拉斯近似（INLA）技术相结合，降低了方法的计算负担。我们通过模拟研究了该方法的性能，并使用该模型预测在肯尼亚家庭集群样本中收集的人体测量响应。

    Prediction is a classic challenge in spatial statistics and the inclusion of spatial covariates can greatly improve predictive performance when incorporated into a model with latent spatial effects. It is desirable to develop flexible regression models that allow for nonlinearities and interactions in the covariate structure. Machine learning models have been suggested in the spatial context, allowing for spatial dependence in the residuals, but fail to provide reliable uncertainty estimates. In this paper, we investigate a novel combination of a Gaussian process spatial model and a Bayesian Additive Regression Tree (BART) model. The computational burden of the approach is reduced by combining Markov chain Monte Carlo (MCMC) with the Integrated Nested Laplace Approximation (INLA) technique. We study the performance of the method via simulations and use the model to predict anthropometric responses, collected via household cluster samples in Kenya.
    

