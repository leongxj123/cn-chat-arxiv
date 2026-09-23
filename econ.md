# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distributional Treatment Effect with Finite Mixture](https://arxiv.org/abs/2403.18503) | 本文提出了使用有限混合模型和控制协变量来处理处理效应分布的方法。 |
| [^2] | [Difference-in-Differences with Unpoolable Data](https://arxiv.org/abs/2403.15910) | 该研究提出了一种创新方法 UN--DID，用于估计具有不可混合数据的差异中的差异，并通过调整附加协变量、多组和错开采纳来提供有关受试者平均处理效应（ATT）的估计。 |
| [^3] | [Beta-Sorted Portfolios.](http://arxiv.org/abs/2208.10974) | 该论文对Beta分类组合投资组合进行了研究，通过将过程形式化为一个由非参数第一步和Beta自适应投资组合构建组成的两步非参数估计器，解释了该估计算法的关键特征，并提供了条件以确保一致性和渐近正态性。 |

# 详细

[^1]: 具有有限混合的分布式处理效应

    Distributional Treatment Effect with Finite Mixture

    [https://arxiv.org/abs/2403.18503](https://arxiv.org/abs/2403.18503)

    本文提出了使用有限混合模型和控制协变量来处理处理效应分布的方法。

    

    处理效应的异质性在评估治疗时非常重要。然而，即使在二元处理的简单情况下，由于我们无法观察到给定个体的已处理潜在结果和未处理潜在结果的基本限制，处理效应的分布也很难确定。本文在潜在结果上假设了一个有限混合模型和一个控制协变量向量，以解决处理内生性，并对每种类型的潜在结果和协变量施加了马尔可夫条件，以确定处理效应分布。有限混合模型的混合权重通过非负矩阵分解算法一致估计，从而使我们能够一致地估计组件分布参数，包括处理效应分布的参数。

    arXiv:2403.18503v1 Announce Type: new  Abstract: Treatment effect heterogeneity is of a great concern when evaluating the treatment. However, even with a simple case of a binary treatment, the distribution of treatment effect is difficult to identify due to the fundamental limitation that we cannot observe both treated potential outcome and untreated potential outcome for a given individual. This paper assumes a finite mixture model on the potential outcomes and a vector of control covariates to address treatment endogeneity and imposes a Markov condition on the potential outcomes and covariates within each type to identify the treatment effect distribution. The mixture weights of the finite mixture model are consistently estimated with a nonnegative matrix factorization algorithm, thus allowing us to consistently estimate the component distribution parameters, including ones for the treatment effect distribution.
    
[^2]: 具有不可混合数据的差异中的差异

    Difference-in-Differences with Unpoolable Data

    [https://arxiv.org/abs/2403.15910](https://arxiv.org/abs/2403.15910)

    该研究提出了一种创新方法 UN--DID，用于估计具有不可混合数据的差异中的差异，并通过调整附加协变量、多组和错开采纳来提供有关受试者平均处理效应（ATT）的估计。

    

    在本研究中，我们确定并放宽了差异中的差异（DID）估计中数据“可混合性”的假设。由于数据隐私问题，往往无法组合来自受试者和对照组的观测数据，因此可混合性不可行。例如，存储在安全设施中的行政健康数据往往无法跨不同司法管辖区组合。我们提出了一种创新方法来估计具有不可混合数据的DID：UN--DID。我们的方法包括对附加协变量、多组和错开采纳进行调整。在没有协变量的情况下，UN--DID和传统DID给出了相同的受试者平均处理效应（ATT）估计。有协变量时，我们通过数学和模拟表明UN--DID和传统DID提供了不同但同样信息丰富的ATT估计。一个实证示例进一步强调了我们方法的实用性。

    arXiv:2403.15910v1 Announce Type: new  Abstract: In this study, we identify and relax the assumption of data "poolability" in difference-in-differences (DID) estimation. Poolability, or the combination of observations from treated and control units into one dataset, is often not possible due to data privacy concerns. For instance, administrative health data stored in secure facilities is often not combinable across jurisdictions. We propose an innovative approach to estimate DID with unpoolable data: UN--DID. Our method incorporates adjustments for additional covariates, multiple groups, and staggered adoption. Without covariates, UN--DID and conventional DID give identical estimates of the average treatment effect on the treated (ATT). With covariates, we show mathematically and through simulations that UN--DID and conventional DID provide different, but equally informative, estimates of the ATT. An empirical example further underscores the utility of our methodology. The UN--DID meth
    
[^3]: Beta分类组合投资组合研究

    Beta-Sorted Portfolios. (arXiv:2208.10974v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2208.10974](http://arxiv.org/abs/2208.10974)

    该论文对Beta分类组合投资组合进行了研究，通过将过程形式化为一个由非参数第一步和Beta自适应投资组合构建组成的两步非参数估计器，解释了该估计算法的关键特征，并提供了条件以确保一致性和渐近正态性。

    

    Beta分类组合投资组合是由与选择的风险因素具有类似协变性的资产组成的，是经济金融领域中分析(条件)预期收益模型的常用工具。尽管使用广泛，但与可比的两步回归等程序相比，对其统计性质知之甚少。我们通过将该过程作为一个由非参数第一步和Beta自适应投资组合构建组成的两步非参数估计器来形式化研究Beta分类组合投资组合回报的性质。我们的框架基于一般数据生成过程上的精确经济和统计假设，从而解释了众所周知的估计算法，并揭示了其关键特征。我们研究了单个截面和随时间聚合（例如总体均值）的Beta分类组合投资组合，提供了确保一致性和渐近正态性的条件，同时还提供了新的均一推断过程，允许不确定性。

    Beta-sorted portfolios -- portfolios comprised of assets with similar covariation to selected risk factors -- are a popular tool in empirical finance to analyze models of (conditional) expected returns. Despite their widespread use, little is known of their statistical properties in contrast to comparable procedures such as two-pass regressions. We formally investigate the properties of beta-sorted portfolio returns by casting the procedure as a two-step nonparametric estimator with a nonparametric first step and a beta-adaptive portfolios construction. Our framework rationalize the well-known estimation algorithm with precise economic and statistical assumptions on the general data generating process and characterize its key features. We study beta-sorted portfolios for both a single cross-section as well as for aggregation over time (e.g., the grand mean), offering conditions that ensure consistency and asymptotic normality along with new uniform inference procedures allowing for unc
    

