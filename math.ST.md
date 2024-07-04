# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits](https://arxiv.org/abs/2402.15171) | 提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。 |
| [^2] | [Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing.](http://arxiv.org/abs/2308.14507) | 本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。 |
| [^3] | [Semiparametric Efficiency Gains from Parametric Restrictions on the Generalized Propensity Score.](http://arxiv.org/abs/2306.04177) | 确定参数模型的广义倾向得分可弱化效率。即使模型很大，知道倾向得分的参数结构还是会提高效率。知道分层结构可以提高效率，特别是当每个层次组件的大小较小时。 |

# 详细

[^1]: 用于随机组合半臂老虎机的协方差自适应最小二乘算法

    Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2402.15171](https://arxiv.org/abs/2402.15171)

    提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。

    

    我们解决了随机组合半臂老虎机问题，其中玩家可以从包含d个基本项的P个子集中进行选择。大多数现有算法（如CUCB、ESCB、OLS-UCB）需要对奖励分布有先验知识，比如子高斯代理-方差的上界，这很难准确估计。在这项工作中，我们设计了OLS-UCB的方差自适应版本，依赖于协方差结构的在线估计。在实际设置中，估计协方差矩阵的系数要容易得多，并且相对于基于代理方差的算法，导致改进的遗憾上界。当协方差系数全为非负时，我们展示了我们的方法有效地利用了半臂反馈，并且可以明显优于老虎机反馈方法，在指数级别P≫d以及P≤d的情况下，这一点并不来自大多数现有分析。

    arXiv:2402.15171v1 Announce Type: new  Abstract: We address the problem of stochastic combinatorial semi-bandits, where a player can select from P subsets of a set containing d base items. Most existing algorithms (e.g. CUCB, ESCB, OLS-UCB) require prior knowledge on the reward distribution, like an upper bound on a sub-Gaussian proxy-variance, which is hard to estimate tightly. In this work, we design a variance-adaptive version of OLS-UCB, relying on an online estimation of the covariance structure. Estimating the coefficients of a covariance matrix is much more manageable in practical settings and results in improved regret upper bounds compared to proxy variance-based algorithms. When covariance coefficients are all non-negative, we show that our approach efficiently leverages the semi-bandit feedback and provably outperforms bandit feedback approaches, not only in exponential regimes where P $\gg$ d but also when P $\le$ d, which is not straightforward from most existing analyses.
    
[^2]: 通过近似传递消息实现结构化广义线性模型的谱估计器

    Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing. (arXiv:2308.14507v1 [math.ST])

    [http://arxiv.org/abs/2308.14507](http://arxiv.org/abs/2308.14507)

    本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。

    

    我们考虑从广义线性模型中的观测中进行参数估计的问题。谱方法是一种简单而有效的估计方法：它通过对观测进行适当预处理得到的矩阵的主特征向量来估计参数。尽管谱估计器被广泛使用，但对于结构化（即独立同分布的高斯和哈尔）设计，目前仅有对谱估计器的严格性能表征以及对数据进行预处理的基本方法可用。相反，实际的设计矩阵具有高度结构化并且表现出非平凡的相关性。为解决这个问题，我们考虑了捕捉测量的非各向同性特性的相关高斯设计，通过特征协方差矩阵Σ进行表示。我们的主要结果是对于这种情况下谱估计器性能的精确渐近分析。然后，可以通过这一结果来确定最优预处理，从而最小化所需样本的数量。

    We consider the problem of parameter estimation from observations given by a generalized linear model. Spectral methods are a simple yet effective approach for estimation: they estimate the parameter via the principal eigenvector of a matrix obtained by suitably preprocessing the observations. Despite their wide use, a rigorous performance characterization of spectral estimators, as well as a principled way to preprocess the data, is available only for unstructured (i.e., i.i.d. Gaussian and Haar) designs. In contrast, real-world design matrices are highly structured and exhibit non-trivial correlations. To address this problem, we consider correlated Gaussian designs which capture the anisotropic nature of the measurements via a feature covariance matrix $\Sigma$. Our main result is a precise asymptotic characterization of the performance of spectral estimators in this setting. This then allows to identify the optimal preprocessing that minimizes the number of samples needed to meanin
    
[^3]: 广义倾向得分的参数限制对半参数效率的提升

    Semiparametric Efficiency Gains from Parametric Restrictions on the Generalized Propensity Score. (arXiv:2306.04177v1 [econ.EM])

    [http://arxiv.org/abs/2306.04177](http://arxiv.org/abs/2306.04177)

    确定参数模型的广义倾向得分可弱化效率。即使模型很大，知道倾向得分的参数结构还是会提高效率。知道分层结构可以提高效率，特别是当每个层次组件的大小较小时。

    

    了解倾向得分可以在估计因果参数时弱化效率，但哪种知识更有用？为了研究这个问题，首先我们推导了在正确指定参数模型的情况下，多值治疗效果的半参数效率下界。然后我们揭示了哪种倾向得分的参数结构即使模型很大也会提高效率。最后，我们将我们开发的一般理论应用于分层实验设置，并发现知道层次结构可以提高效率，特别是当每个层次组件的大小较小时。

    Knowledge of the propensity score weakly improves efficiency when estimating causal parameters, but what kind of knowledge is more useful? To examine this, we first derive the semiparametric efficiency bound of multivalued treatment effects when the propensity score is correctly specified by a parametric model. We then reveal which parametric structure on the propensity score enhances the efficiency even when the the model is large. Finally, we apply the general theory we develop to a stratified experiment setup and find that knowing the strata improves the efficiency, especially when the size of each stratum component is small.
    

