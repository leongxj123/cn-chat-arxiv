# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference for Rank-Rank Regressions.](http://arxiv.org/abs/2310.15512) | 本文研究了等级回归中常用的方差估计器在估计OLS估计器的渐进方差时的不一致性问题，并提出了一种一致估计器。应用新的推论方法在三个经验研究中发现，基于正确方差的估计器的置信区间可能欠精确。 |
| [^2] | [Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing.](http://arxiv.org/abs/2308.14507) | 本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。 |

# 详细

[^1]: 推论用于等级回归

    Inference for Rank-Rank Regressions. (arXiv:2310.15512v1 [econ.EM])

    [http://arxiv.org/abs/2310.15512](http://arxiv.org/abs/2310.15512)

    本文研究了等级回归中常用的方差估计器在估计OLS估计器的渐进方差时的不一致性问题，并提出了一种一致估计器。应用新的推论方法在三个经验研究中发现，基于正确方差的估计器的置信区间可能欠精确。

    

    在等级回归中，斜率系数是衡量代际流动性的常用指标，例如在子女收入等级与父母收入等级回归中。本文首先指出，常用的方差估计器如同方差估计器或鲁棒方差估计器未能一致估计OLS估计器在等级回归中的渐进方差。我们表明，这些估计器的概率极限可能过大或过小，取决于子女收入和父母收入的联合分布函数的形状。其次，我们导出了等级回归的一般渐进理论，并提供了OLS估计器渐进方差的一致估计器。然后，我们将渐进理论扩展到其他经验工作中涉及等级的回归。最后，我们将新的推论方法应用于三个经验研究。我们发现，基于正确方差的估计器的置信区间有时可能欠精确。

    Slope coefficients in rank-rank regressions are popular measures of intergenerational mobility, for instance in regressions of a child's income rank on their parent's income rank. In this paper, we first point out that commonly used variance estimators such as the homoskedastic or robust variance estimators do not consistently estimate the asymptotic variance of the OLS estimator in a rank-rank regression. We show that the probability limits of these estimators may be too large or too small depending on the shape of the copula of child and parent incomes. Second, we derive a general asymptotic theory for rank-rank regressions and provide a consistent estimator of the OLS estimator's asymptotic variance. We then extend the asymptotic theory to other regressions involving ranks that have been used in empirical work. Finally, we apply our new inference methods to three empirical studies. We find that the confidence intervals based on estimators of the correct variance may sometimes be sub
    
[^2]: 通过近似传递消息实现结构化广义线性模型的谱估计器

    Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing. (arXiv:2308.14507v1 [math.ST])

    [http://arxiv.org/abs/2308.14507](http://arxiv.org/abs/2308.14507)

    本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。

    

    我们考虑从广义线性模型中的观测中进行参数估计的问题。谱方法是一种简单而有效的估计方法：它通过对观测进行适当预处理得到的矩阵的主特征向量来估计参数。尽管谱估计器被广泛使用，但对于结构化（即独立同分布的高斯和哈尔）设计，目前仅有对谱估计器的严格性能表征以及对数据进行预处理的基本方法可用。相反，实际的设计矩阵具有高度结构化并且表现出非平凡的相关性。为解决这个问题，我们考虑了捕捉测量的非各向同性特性的相关高斯设计，通过特征协方差矩阵Σ进行表示。我们的主要结果是对于这种情况下谱估计器性能的精确渐近分析。然后，可以通过这一结果来确定最优预处理，从而最小化所需样本的数量。

    We consider the problem of parameter estimation from observations given by a generalized linear model. Spectral methods are a simple yet effective approach for estimation: they estimate the parameter via the principal eigenvector of a matrix obtained by suitably preprocessing the observations. Despite their wide use, a rigorous performance characterization of spectral estimators, as well as a principled way to preprocess the data, is available only for unstructured (i.e., i.i.d. Gaussian and Haar) designs. In contrast, real-world design matrices are highly structured and exhibit non-trivial correlations. To address this problem, we consider correlated Gaussian designs which capture the anisotropic nature of the measurements via a feature covariance matrix $\Sigma$. Our main result is a precise asymptotic characterization of the performance of spectral estimators in this setting. This then allows to identify the optimal preprocessing that minimizes the number of samples needed to meanin
    

