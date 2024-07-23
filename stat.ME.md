# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Causal Graph Discovery Using Large Language Models](https://rss.arxiv.org/abs/2402.01207) | 提出了一个新的框架，利用大型语言模型进行高效的因果图发现，采用了广度优先搜索方法，只需要线性数量的查询，同时能轻松结合观察数据以提高性能，具有高效性和数据效率，并在真实因果图上取得了最先进的结果，展示了其在不同领域的广泛适用性潜力。 |
| [^2] | [Machine Learning Assisted Adjustment Boosts Inferential Efficiency of Randomized Controlled Trials](https://arxiv.org/abs/2403.03058) | 该研究提出了一种机器学习辅助调整的推断程序，可以提高随机对照试验的推断效率，并在样本量和成本方面有显著的优势。 |
| [^3] | [Resolution of Simpson's paradox via the common cause principle](https://arxiv.org/abs/2403.00957) | 通过对共同原因$C$进行条件设定，解决了辛普森悖论，推广了悖论，并表明在二元共同原因$C$上进行条件设定的关联方向与原始$B$上进行条件设定相同 |
| [^4] | [Information-Enriched Selection of Stationary and Non-Stationary Autoregressions using the Adaptive Lasso](https://arxiv.org/abs/2402.16580) | 该研究提出了一种新的方法，在估计自回归模型时引入潜在非平稳回归变量的权重，获得了理论和实证结果显示这种方法在检测平稳性方面的优势。 |
| [^5] | [Identifying Causal Effects using Instrumental Time Series: Nuisance IV and Correcting for the Past](https://arxiv.org/abs/2203.06056) | 本文考虑了在时间序列模型中进行 IV 回归的困难，提出了一种用于构建识别方程的方法，以实现对时间序列数据中因果效应的一致性参数估计。 |
| [^6] | [Dynamical System Identification, Model Selection and Model Uncertainty Quantification by Bayesian Inference.](http://arxiv.org/abs/2401.16943) | 本研究提出了一种基于贝叶斯推理的动力系统识别方法，可以估计模型系数，进行模型排序和模型不确定性量化，并展示了与其他算法的比较结果。 |
| [^7] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |
| [^8] | [High-Dimensional Bayesian Structure Learning in Gaussian Graphical Models using Marginal Pseudo-Likelihood.](http://arxiv.org/abs/2307.00127) | 该论文提出了两种创新的搜索算法，在高维图结构学习中使用边际伪似然函数解决计算复杂性问题，并且能够在短时间内生成可靠的估计。该方法提供了R软件包BDgraph的代码实现。 |
| [^9] | [Covariate Adjustment in Stratified Experiments.](http://arxiv.org/abs/2302.03687) | 本文研究了样本分层实验中的协变量调整方法，并发现对于分层设计，传统回归估计量通常是低效的。根据这一结果，我们推导了针对给定分层的渐近最优线性协变量调整方法，并构建了几个可行的大样本估计器。 |

# 详细

[^1]: 使用大型语言模型的高效因果图发现

    Efficient Causal Graph Discovery Using Large Language Models

    [https://rss.arxiv.org/abs/2402.01207](https://rss.arxiv.org/abs/2402.01207)

    提出了一个新的框架，利用大型语言模型进行高效的因果图发现，采用了广度优先搜索方法，只需要线性数量的查询，同时能轻松结合观察数据以提高性能，具有高效性和数据效率，并在真实因果图上取得了最先进的结果，展示了其在不同领域的广泛适用性潜力。

    

    我们提出了一个新的框架，利用LLMs进行完整的因果图发现。之前基于LLM的方法采用了成对查询的方法，但这需要二次查询的数量，对于较大的因果图来说很快变得不可行。相反，提出的框架采用了广度优先搜索（BFS）的方法，只需要线性数量的查询。我们还展示了当有所观察数据可用时，提出的方法可以轻松地进行结合以提高性能。除了更具时间和数据效率外，提出的框架在不同大小的真实因果图上取得了最先进的结果。结果证明了提出方法在发现因果关系方面的有效性和效率，展示了其在不同领域的因果图发现任务中的广泛适用性潜力。

    We propose a novel framework that leverages LLMs for full causal graph discovery. While previous LLM-based methods have used a pairwise query approach, this requires a quadratic number of queries which quickly becomes impractical for larger causal graphs. In contrast, the proposed framework uses a breadth-first search (BFS) approach which allows it to use only a linear number of queries. We also show that the proposed method can easily incorporate observational data when available, to improve performance. In addition to being more time and data-efficient, the proposed framework achieves state-of-the-art results on real-world causal graphs of varying sizes. The results demonstrate the effectiveness and efficiency of the proposed method in discovering causal relationships, showcasing its potential for broad applicability in causal graph discovery tasks across different domains.
    
[^2]: 机器学习辅助调整提升随机对照试验的推断效率

    Machine Learning Assisted Adjustment Boosts Inferential Efficiency of Randomized Controlled Trials

    [https://arxiv.org/abs/2403.03058](https://arxiv.org/abs/2403.03058)

    该研究提出了一种机器学习辅助调整的推断程序，可以提高随机对照试验的推断效率，并在样本量和成本方面有显著的优势。

    

    在这项工作中，我们提出了一种新的推断程序，该程序采用了基于机器学习的调整方法，用于随机对照试验。该方法是在罗森鲍姆的基于协变量调整的随机实验的确切检验框架下开发的。通过大量的模拟实验，我们展示了所提出的方法可以稳健地控制第一类错误，并可以提高随机对照试验(RCT)的推断效率。这一优势在一个真实案例中进一步得到了证明。所提出方法的简单性和稳健性使其成为一种竞争性候选作为RCT的常规推断程序，特别是当基线协变量的数量较多，且预计存在非线性关联或协变量之间的交互作用时。其应用可以显著降低RCT的所需样本量和成本，例如三期临床试验。

    arXiv:2403.03058v1 Announce Type: cross  Abstract: In this work, we proposed a novel inferential procedure assisted by machine learning based adjustment for randomized control trials. The method was developed under the Rosenbaum's framework of exact tests in randomized experiments with covariate adjustments. Through extensive simulation experiments, we showed the proposed method can robustly control the type I error and can boost the inference efficiency for a randomized controlled trial (RCT). This advantage was further demonstrated in a real world example. The simplicity and robustness of the proposed method makes it a competitive candidate as a routine inference procedure for RCTs, especially when the number of baseline covariates is large, and when nonlinear association or interaction among covariates is expected. Its application may remarkably reduce the required sample size and cost of RCTs, such as phase III clinical trials.
    
[^3]: 利用共因原则解决辛普森悖论

    Resolution of Simpson's paradox via the common cause principle

    [https://arxiv.org/abs/2403.00957](https://arxiv.org/abs/2403.00957)

    通过对共同原因$C$进行条件设定，解决了辛普森悖论，推广了悖论，并表明在二元共同原因$C$上进行条件设定的关联方向与原始$B$上进行条件设定相同

    

    辛普森悖论是建立两个事件$a_1$和$a_2$之间的概率关联时的障碍，给定第三个（潜在的）随机变量$B$。我们关注的情景是随机变量$A$（汇总了$a_1$、$a_2$及其补集）和$B$有一个可能未被观察到的共同原因$C$。或者，我们可以假设$C$将$A$从$B$中筛选出去。对于这种情况，正确的$a_1$和$a_2$之间的关联应该通过对$C$进行条件设定来定义。这一设置将原始辛普森悖论推广了。现在它的两个相互矛盾的选项简单地指的是两个特定且不同的原因$C$。我们表明，如果$B$和$C$是二进制的，$A$是四进制的（对于有效的辛普森悖论来说是最小且最常见的情况），在任何二元共同原因$C$上进行条件设定将建立与在原始$B$上进行条件设定相同的$a_1$和$a_2$之间的关联方向。

    arXiv:2403.00957v1 Announce Type: cross  Abstract: Simpson's paradox is an obstacle to establishing a probabilistic association between two events $a_1$ and $a_2$, given the third (lurking) random variable $B$. We focus on scenarios when the random variables $A$ (which combines $a_1$, $a_2$, and their complements) and $B$ have a common cause $C$ that need not be observed. Alternatively, we can assume that $C$ screens out $A$ from $B$. For such cases, the correct association between $a_1$ and $a_2$ is to be defined via conditioning over $C$. This set-up generalizes the original Simpson's paradox. Now its two contradicting options simply refer to two particular and different causes $C$. We show that if $B$ and $C$ are binary and $A$ is quaternary (the minimal and the most widespread situation for valid Simpson's paradox), the conditioning over any binary common cause $C$ establishes the same direction of the association between $a_1$ and $a_2$ as the conditioning over $B$ in the original
    
[^4]: 使用自适应Lasso选择平稳和非平稳自回归模型的信息增强方法

    Information-Enriched Selection of Stationary and Non-Stationary Autoregressions using the Adaptive Lasso

    [https://arxiv.org/abs/2402.16580](https://arxiv.org/abs/2402.16580)

    该研究提出了一种新的方法，在估计自回归模型时引入潜在非平稳回归变量的权重，获得了理论和实证结果显示这种方法在检测平稳性方面的优势。

    

    我们提出了一种新方法，通过使用自适应Lasso，在一致和奥拉克尔效率估计自回归模型时引入一个潜在非平稳回归变量的权重。增强的权重建立在一个统计量上，该统计量利用OLS估计器在时间序列回归中的概率顺序不同的特点，当积分程度不同时。我们在选择$\ell_1$惩罚参数时提供了理论结果，证明了我们的方法在检测平稳性方面的优势。蒙特卡洛证据表明，我们的提议优于使用OLS基础权重，正如Kock建议的那样。我们将修改后的估计器应用于欧元推出后德国通货膨胀率的模型选择。结果表明，能源商品价格通货膨胀和整体通货膨胀最好由平稳自回归模型描述。

    arXiv:2402.16580v1 Announce Type: cross  Abstract: We propose a novel approach to elicit the weight of a potentially non-stationary regressor in the consistent and oracle-efficient estimation of autoregressive models using the adaptive Lasso. The enhanced weight builds on a statistic that exploits distinct orders in probability of the OLS estimator in time series regressions when the degree of integration differs. We provide theoretical results on the benefit of our approach for detecting stationarity when a tuning criterion selects the $\ell_1$ penalty parameter. Monte Carlo evidence shows that our proposal is superior to using OLS-based weights, as suggested by Kock [Econom. Theory, 32, 2016, 243-259]. We apply the modified estimator to model selection for German inflation rates after the introduction of the Euro. The results indicate that energy commodity price inflation and headline inflation are best described by stationary autoregressions.
    
[^5]: 使用工具时间序列识别因果效应：无关 IV 和纠正历史

    Identifying Causal Effects using Instrumental Time Series: Nuisance IV and Correcting for the Past

    [https://arxiv.org/abs/2203.06056](https://arxiv.org/abs/2203.06056)

    本文考虑了在时间序列模型中进行 IV 回归的困难，提出了一种用于构建识别方程的方法，以实现对时间序列数据中因果效应的一致性参数估计。

    

    仪器变量（IV）回归依赖于工具来推断观测数据中的因果效应，其中存在未观测的混淆因素。我们考虑在时间序列模型中进行 IV 回归，例如矢量自回归（VAR）过程。直接应用独立同分布技术通常不一致，因为它们不能正确调整过去的依赖关系。本文概述了由于时间结构而引起的困难，并提出了用于构建可用于时间序列数据中因果效应一致参数估计的确认方程的方法。一种方法使用额外的无关协变量来获得可识别性（即使在独立同分布情况下也是有趣的想法）。我们进一步提出了一个图边缘化框架，允许我们以原则性的方式对时间序列应用无关 IV 和其他 IV 方法。我们的方法利用了全局马尔可夫性质的一个版本。

    arXiv:2203.06056v2 Announce Type: replace-cross  Abstract: Instrumental variable (IV) regression relies on instruments to infer causal effects from observational data with unobserved confounding. We consider IV regression in time series models, such as vector auto-regressive (VAR) processes. Direct applications of i.i.d. techniques are generally inconsistent as they do not correctly adjust for dependencies in the past. In this paper, we outline the difficulties that arise due to time structure and propose methodology for constructing identifying equations that can be used for consistent parametric estimation of causal effects in time series data. One method uses extra nuisance covariates to obtain identifiability (an idea that can be of interest even in the i.i.d. case). We further propose a graph marginalization framework that allows us to apply nuisance IV and other IV methods in a principled way to time series. Our methods make use of a version of the global Markov property, which w
    
[^6]: 动力系统识别、模型选择和贝叶斯推理中的模型不确定性量化

    Dynamical System Identification, Model Selection and Model Uncertainty Quantification by Bayesian Inference. (arXiv:2401.16943v1 [stat.ME])

    [http://arxiv.org/abs/2401.16943](http://arxiv.org/abs/2401.16943)

    本研究提出了一种基于贝叶斯推理的动力系统识别方法，可以估计模型系数，进行模型排序和模型不确定性量化，并展示了与其他算法的比较结果。

    

    本研究提出了一种基于贝叶斯最大后验概率 (MAP) 框架的动力系统识别方法，用于从时间序列数据中恢复系统模型。实验证明这等价于广义的零阶 Tikhonov 正则化，通过负对数似然和先验分布来合理选择残差和正则化项。除了估计模型系数外，贝叶斯解释还提供了完整的贝叶斯推理工具，包括模型排序、模型不确定性量化和未知超参数的估计。通过应用于带有噪声的几个动力系统，比较了两种贝叶斯算法，即联合最大后验概率 (JMAP) 和变分贝叶斯近似 (VBA)，与流行的阈值最小二乘回归算法SINDy。对于多元高斯似然和先验分布，

    This study presents a Bayesian maximum \textit{a~posteriori} (MAP) framework for dynamical system identification from time-series data. This is shown to be equivalent to a generalized zeroth-order Tikhonov regularization, providing a rational justification for the choice of the residual and regularization terms, respectively, from the negative logarithms of the likelihood and prior distributions. In addition to the estimation of model coefficients, the Bayesian interpretation gives access to the full apparatus for Bayesian inference, including the ranking of models, the quantification of model uncertainties and the estimation of unknown (nuisance) hyperparameters. Two Bayesian algorithms, joint maximum \textit{a~posteriori} (JMAP) and variational Bayesian approximation (VBA), are compared to the popular SINDy algorithm for thresholded least-squares regression, by application to several dynamical systems with added noise. For multivariate Gaussian likelihood and prior distributions, the
    
[^7]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    
[^8]: 高维贝叶斯高斯图模型中的结构学习方法——利用边际伪似然函数

    High-Dimensional Bayesian Structure Learning in Gaussian Graphical Models using Marginal Pseudo-Likelihood. (arXiv:2307.00127v1 [stat.ME])

    [http://arxiv.org/abs/2307.00127](http://arxiv.org/abs/2307.00127)

    该论文提出了两种创新的搜索算法，在高维图结构学习中使用边际伪似然函数解决计算复杂性问题，并且能够在短时间内生成可靠的估计。该方法提供了R软件包BDgraph的代码实现。

    

    高斯图模型以图形形式描绘了多元正态分布中变量之间的条件依赖关系。这篇论文介绍了两种创新的搜索算法，利用边际伪似然函数来应对高维图结构学习中的计算复杂性问题。这些方法可以在标准计算机上在几分钟内快速生成对包含1000个变量的问题的可靠估计。对于对实际应用感兴趣的人，支持这种新方法的代码通过R软件包BDgraph提供。

    Gaussian graphical models depict the conditional dependencies between variables within a multivariate normal distribution in a graphical format. The identification of these graph structures is an area known as structure learning. However, when utilizing Bayesian methodologies in structure learning, computational complexities can arise, especially with high-dimensional graphs surpassing 250 nodes. This paper introduces two innovative search algorithms that employ marginal pseudo-likelihood to address this computational challenge. These methods can swiftly generate reliable estimations for problems encompassing 1000 variables in just a few minutes on standard computers. For those interested in practical applications, the code supporting this new approach is made available through the R package BDgraph.
    
[^9]: 样本分层实验中的协变量调整

    Covariate Adjustment in Stratified Experiments. (arXiv:2302.03687v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.03687](http://arxiv.org/abs/2302.03687)

    本文研究了样本分层实验中的协变量调整方法，并发现对于分层设计，传统回归估计量通常是低效的。根据这一结果，我们推导了针对给定分层的渐近最优线性协变量调整方法，并构建了几个可行的大样本估计器。

    

    本文研究了在样本分层实验中对平均处理效应进行协变量调整的方法。我们在一个通用的框架中工作，包括匹配元组设计、粗分层设计和完全随机化设计作为特例。已知对于完全随机化设计，协变量调整与处理-协变量交互项可以弱化效率。然而，我们发现，对于分层设计，这种回归估计量通常是低效的，甚至可能相对于未调整的基准估计量增加估计方差。在此结果的基础上，我们推导出针对给定分层的渐近最优线性协变量调整。我们构建了几个可行的估计器，以使大样本中实现这种高效调整。例如，在匹配对的特例中，包括处理、协变量和对固定效应的回归在渐近上是最优的。我们还提供了新颖的渐近精确推断方法。

    This paper studies covariate adjusted estimation of the average treatment effect in stratified experiments. We work in a general framework that includes matched tuples designs, coarse stratification, and complete randomization as special cases. Regression adjustment with treatment-covariate interactions is known to weakly improve efficiency for completely randomized designs. By contrast, we show that for stratified designs such regression estimators are generically inefficient, potentially even increasing estimator variance relative to the unadjusted benchmark. Motivated by this result, we derive the asymptotically optimal linear covariate adjustment for a given stratification. We construct several feasible estimators that implement this efficient adjustment in large samples. In the special case of matched pairs, for example, the regression including treatment, covariates, and pair fixed effects is asymptotically optimal. We also provide novel asymptotically exact inference methods tha
    

