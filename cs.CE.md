# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian approach to Gaussian process regression with uncertain inputs.](http://arxiv.org/abs/2305.11586) | 本文提出了一种新的高斯过程回归技术，通过贝叶斯方法将输入数据的不确定性纳入回归模型预测中。在数值实验中展示了该方法具有普适性和不错的表现。 |

# 详细

[^1]: 高斯过程回归的贝叶斯方法中融入不确定输入

    Bayesian approach to Gaussian process regression with uncertain inputs. (arXiv:2305.11586v1 [cs.LG])

    [http://arxiv.org/abs/2305.11586](http://arxiv.org/abs/2305.11586)

    本文提出了一种新的高斯过程回归技术，通过贝叶斯方法将输入数据的不确定性纳入回归模型预测中。在数值实验中展示了该方法具有普适性和不错的表现。

    

    传统高斯过程回归仅假设模型观测数据的输出具有噪声。然而，在许多科学和工程应用中，由于建模假设、测量误差等因素，观测数据的输入位置可能也存在不确定性。在本文中，我们提出了一种贝叶斯方法，将输入数据的可变性融入到高斯过程回归中。考虑两种可观测量——具有固定输入的噪声污染输出和具有先验分布定义的不确定输入，通过贝叶斯框架估计后验分布以推断不确定的数据位置。然后，利用边际化方法将这些输入的量化不确定性纳入高斯过程预测中。通过几个数值实验，展示了这种新回归技术的有效性，在其中观察到不同水平输入数据不确定性下的普适良好表现。

    Conventional Gaussian process regression exclusively assumes the existence of noise in the output data of model observations. In many scientific and engineering applications, however, the input locations of observational data may also be compromised with uncertainties owing to modeling assumptions, measurement errors, etc. In this work, we propose a Bayesian method that integrates the variability of input data into Gaussian process regression. Considering two types of observables -- noise-corrupted outputs with fixed inputs and those with prior-distribution-defined uncertain inputs, a posterior distribution is estimated via a Bayesian framework to infer the uncertain data locations. Thereafter, such quantified uncertainties of inputs are incorporated into Gaussian process predictions by means of marginalization. The effectiveness of this new regression technique is demonstrated through several numerical examples, in which a consistently good performance of generalization is observed, w
    

