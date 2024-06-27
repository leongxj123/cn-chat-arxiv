# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Hierarchical Causal Models.](http://arxiv.org/abs/2401.05330) | 提出了一种分层因果模型来解决关于分层数据的因果问题，通过添加内部板来扩展结构因果模型和因果图模型。发现分层数据可以实现因果识别，即使使用非分层数据是不可能的。开发了用于分层数据的估计技术。 |
| [^3] | [A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation.](http://arxiv.org/abs/2306.16297) | 这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。 |
| [^4] | [STEEL: Singularity-aware Reinforcement Learning.](http://arxiv.org/abs/2301.13152) | 这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 分层因果模型

    Hierarchical Causal Models. (arXiv:2401.05330v1 [stat.ME])

    [http://arxiv.org/abs/2401.05330](http://arxiv.org/abs/2401.05330)

    提出了一种分层因果模型来解决关于分层数据的因果问题，通过添加内部板来扩展结构因果模型和因果图模型。发现分层数据可以实现因果识别，即使使用非分层数据是不可能的。开发了用于分层数据的估计技术。

    

    科学家们经常想要从分层数据中学习因果关系，这些数据是从嵌套在单位内部的子单元收集的。比如学校中的学生、病人的细胞或州中的城市。在这种情况下，单位级变量（例如每个学校的预算）可能会影响子单位级变量（例如每个学校每个学生的考试成绩），反之亦然。为了解决关于分层数据的因果问题，我们提出了分层因果模型，它通过添加内部板来扩展结构因果模型和因果图模型。我们开发了一种用于分层因果模型的通用图形识别技术，该技术扩展了do-计算。我们发现许多情况下，即使使用非分层数据是不可能的，分层数据也可以实现因果识别，也就是说，如果我们只有子单位级变量的单位级汇总（例如学校的平均考试成绩，而不是每个学生的成绩）。我们开发了用于分层数据的估计技术。

    Scientists often want to learn about cause and effect from hierarchical data, collected from subunits nested inside units. Consider students in schools, cells in patients, or cities in states. In such settings, unit-level variables (e.g. each school's budget) may affect subunit-level variables (e.g. the test scores of each student in each school) and vice versa. To address causal questions with hierarchical data, we propose hierarchical causal models, which extend structural causal models and causal graphical models by adding inner plates. We develop a general graphical identification technique for hierarchical causal models that extends do-calculus. We find many situations in which hierarchical data can enable causal identification even when it would be impossible with non-hierarchical data, that is, if we had only unit-level summaries of subunit-level variables (e.g. the school's average test score, rather than each student's score). We develop estimation techniques for hierarchical 
    
[^3]: 一种用于评估时变调节因素的因果偏离效应估计的元学习方法

    A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation. (arXiv:2306.16297v1 [stat.ME])

    [http://arxiv.org/abs/2306.16297](http://arxiv.org/abs/2306.16297)

    这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。

    

    可穿戴技术和智能手机提供的数字化健康干预的双重革命显著增加了移动健康（mHealth）干预在各个健康科学领域的可及性和采纳率。顺序随机实验称为微随机试验（MRTs）已经越来越受欢迎，用于实证评估这些mHealth干预组成部分的有效性。MRTs产生了一类新的因果估计量，称为“因果偏离效应”，使健康科学家能够评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。然而，目前用于估计因果偏离效应的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型。虽然机器学习算法在自动特征构建方面具有优势，但其朴素应用导致了问题。

    Twin revolutions in wearable technologies and smartphone-delivered digital health interventions have significantly expanded the accessibility and uptake of mobile health (mHealth) interventions across various health science domains. Sequentially randomized experiments called micro-randomized trials (MRTs) have grown in popularity to empirically evaluate the effectiveness of these mHealth intervention components. MRTs have given rise to a new class of causal estimands known as "causal excursion effects", which enable health scientists to assess how intervention effectiveness changes over time or is moderated by individual characteristics, context, or responses in the past. However, current data analysis methods for estimating causal excursion effects require pre-specified features of the observed high-dimensional history to construct a working model of an important nuisance parameter. While machine learning algorithms are ideal for automatic feature construction, their naive application
    
[^4]: STEEL: 奇异性感知的强化学习

    STEEL: Singularity-aware Reinforcement Learning. (arXiv:2301.13152v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.13152](http://arxiv.org/abs/2301.13152)

    这篇论文介绍了一种新的批量强化学习算法STEEL，在具有连续状态和行动的无限时马尔可夫决策过程中，不依赖于绝对连续假设，通过最大均值偏差和分布鲁棒优化确保异常情况下的性能。

    

    批量强化学习旨在利用预先收集的数据，在动态环境中找到最优策略，以最大化期望总回报。然而，几乎所有现有算法都依赖于目标策略诱导的分布绝对连续假设，以便通过变换测度使用批量数据来校准目标策略。本文提出了一种新的批量强化学习算法，不需要在具有连续状态和行动的无限时马尔可夫决策过程中绝对连续性假设。我们称这个算法为STEEL：SingulariTy-awarE rEinforcement Learning。我们的算法受到关于离线评估的新误差分析的启发，其中我们使用了最大均值偏差，以及带有分布鲁棒优化的策略定向误差评估方法，以确保异常情况下的性能，并提出了一种用于处理奇异情况的定向算法。

    Batch reinforcement learning (RL) aims at leveraging pre-collected data to find an optimal policy that maximizes the expected total rewards in a dynamic environment. Nearly all existing algorithms rely on the absolutely continuous assumption on the distribution induced by target policies with respect to the data distribution, so that the batch data can be used to calibrate target policies via the change of measure. However, the absolute continuity assumption could be violated in practice (e.g., no-overlap support), especially when the state-action space is large or continuous. In this paper, we propose a new batch RL algorithm without requiring absolute continuity in the setting of an infinite-horizon Markov decision process with continuous states and actions. We call our algorithm STEEL: SingulariTy-awarE rEinforcement Learning. Our algorithm is motivated by a new error analysis on off-policy evaluation, where we use maximum mean discrepancy, together with distributionally robust opti
    

