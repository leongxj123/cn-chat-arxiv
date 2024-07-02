# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Analyzing the Reporting Error of Public Transport Trips in the Danish National Travel Survey Using Smart Card Data.](http://arxiv.org/abs/2308.01198) | 本研究使用丹麦智能卡数据和全国出行调查数据，发现在公共交通用户的时间报告中存在中位数为11.34分钟的报告误差。 |
| [^2] | [A Deep Learning Approach for Overall Survival Analysis with Missing Values.](http://arxiv.org/abs/2307.11465) | 提出了一个深度学习模型，通过有效利用被审查和未被审查病人的信息，预测非小细胞肺癌（NSCLC）病人的整体生存。 |
| [^3] | [Bayesian Safety Validation for Black-Box Systems.](http://arxiv.org/abs/2305.02449) | 本文提出了一种名为贝叶斯安全验证的算法，将黑盒安全验证问题转化为贝叶斯优化问题。该算法通过概率代理模型拟合快速预测故障，利用重要性采样估计操作域内的故障概率，从而实现了对高维、危险、计算昂贵的系统的高效估计。 |
| [^4] | [Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment.](http://arxiv.org/abs/2208.13065) | 本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。 |
| [^5] | [Binary response model with many weak instruments.](http://arxiv.org/abs/2201.04811) | 本文提供了一种使用控制函数方法和正则化方案来获得更好内生二进制响应模型估计结果的方法，并应用于研究家庭收入对大学完成的影响。 |

# 详细

[^1]: 使用智能卡数据分析丹麦全国出行调查中公共交通的报告误差

    Analyzing the Reporting Error of Public Transport Trips in the Danish National Travel Survey Using Smart Card Data. (arXiv:2308.01198v1 [stat.AP])

    [http://arxiv.org/abs/2308.01198](http://arxiv.org/abs/2308.01198)

    本研究使用丹麦智能卡数据和全国出行调查数据，发现在公共交通用户的时间报告中存在中位数为11.34分钟的报告误差。

    

    家庭出行调查已经用于数十年来收集个人和家庭的出行行为。然而，自我报告的调查存在回忆偏差，因为受访者可能难以准确回忆和报告他们的活动。本研究通过将两个数据源的连续五年数据在个体层面进行匹配（即丹麦国家出行调查和丹麦智能卡系统），来研究全国范围内家庭出行调查中公共交通用户的时间报告误差。调查受访者与智能卡数据中的旅行卡进行匹配，匹配仅基于受访者声明的时空旅行行为。大约70%的受访者成功与智能卡进行匹配。研究结果显示，中位数的时间报告误差为11.34分钟，四分位范围为28.14分钟。此外，进行了统计分析来探索调查响应者特征和报告误差之间的关系。

    Household travel surveys have been used for decades to collect individuals and households' travel behavior. However, self-reported surveys are subject to recall bias, as respondents might struggle to recall and report their activities accurately. This study addresses examines the time reporting error of public transit users in a nationwide household travel survey by matching, at the individual level, five consecutive years of data from two sources, namely the Danish National Travel Survey (TU) and the Danish Smart Card system (Rejsekort). Survey respondents are matched with travel cards from the Rejsekort data solely based on the respondents' declared spatiotemporal travel behavior. Approximately, 70% of the respondents were successfully matched with Rejsekort travel cards. The findings reveal a median time reporting error of 11.34 minutes, with an Interquartile Range of 28.14 minutes. Furthermore, a statistical analysis was performed to explore the relationships between the survey res
    
[^2]: 一种用于具有缺失值的整体生存分析的深度学习方法

    A Deep Learning Approach for Overall Survival Analysis with Missing Values. (arXiv:2307.11465v1 [cs.LG])

    [http://arxiv.org/abs/2307.11465](http://arxiv.org/abs/2307.11465)

    提出了一个深度学习模型，通过有效利用被审查和未被审查病人的信息，预测非小细胞肺癌（NSCLC）病人的整体生存。

    

    人工智能可以应用于肺癌研究，尤其是非小细胞肺癌（NSCLC），这是一个具有挑战性的领域。对于病人状态的整体生存（OS）是一个重要指标，可以帮助识别生存概率不同的亚组，从而实现个体化治疗和改善整体生存率。在这个分析中，需要考虑两个挑战。首先，很少有研究能够有效利用每个病人的可用信息，利用未被审查的（即死亡）和被审查的（即幸存者）病人的信息，也要考虑到死亡时间。其次，不完整数据处理是医学领域常见的问题。这个问题通常通过使用插补方法来解决。我们的目标是提出一个能够克服这些限制的人工智能模型，能够从被审查和未被审查的病人及其可用特征中有效学习，预测NSCLC病人的OS。

    One of the most challenging fields where Artificial Intelligence (AI) can be applied is lung cancer research, specifically non-small cell lung cancer (NSCLC). In particular, overall survival (OS) is a vital indicator of patient status, helping to identify subgroups with diverse survival probabilities, enabling tailored treatment and improved OS rates. In this analysis, there are two challenges to take into account. First, few studies effectively exploit the information available from each patient, leveraging both uncensored (i.e., dead) and censored (i.e., survivors) patients, considering also the death times. Second, the handling of incomplete data is a common issue in the medical field. This problem is typically tackled through the use of imputation methods. Our objective is to present an AI model able to overcome these limits, effectively learning from both censored and uncensored patients and their available features, for the prediction of OS for NSCLC patients. We present a novel 
    
[^3]: 黑盒系统的贝叶斯安全验证

    Bayesian Safety Validation for Black-Box Systems. (arXiv:2305.02449v1 [cs.LG])

    [http://arxiv.org/abs/2305.02449](http://arxiv.org/abs/2305.02449)

    本文提出了一种名为贝叶斯安全验证的算法，将黑盒安全验证问题转化为贝叶斯优化问题。该算法通过概率代理模型拟合快速预测故障，利用重要性采样估计操作域内的故障概率，从而实现了对高维、危险、计算昂贵的系统的高效估计。

    

    对于安全关键系统准确估计故障概率对认证至关重要。由于高维输入空间、危险测试场景和计算昂贵的仿真器，估计通常具有挑战性，因此研究高效估计技术十分重要。本文将黑盒安全验证问题重新定义为贝叶斯优化问题，并引入一种算法——贝叶斯安全验证，该算法通过迭代拟合概率代理模型来高效预测故障。该算法旨在搜索故障、计算最可能的故障，并利用重要性采样估计操作域内的故障概率。我们引入了三种采集函数，重点是通过覆盖设计空间、优化解析派生的故障边界和采样预测的故障区域来减少不确定性。主要涉及只输出二进制指标的系统。

    Accurately estimating the probability of failure for safety-critical systems is important for certification. Estimation is often challenging due to high-dimensional input spaces, dangerous test scenarios, and computationally expensive simulators; thus, efficient estimation techniques are important to study. This work reframes the problem of black-box safety validation as a Bayesian optimization problem and introduces an algorithm, Bayesian safety validation, that iteratively fits a probabilistic surrogate model to efficiently predict failures. The algorithm is designed to search for failures, compute the most-likely failure, and estimate the failure probability over an operating domain using importance sampling. We introduce a set of three acquisition functions that focus on reducing uncertainty by covering the design space, optimizing the analytically derived failure boundaries, and sampling the predicted failure regions. Mainly concerned with systems that only output a binary indicat
    
[^4]: 改善运营经济学：基于双层 MIP 的闭环预测优化框架来预测机组组合的操作计划

    Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment. (arXiv:2208.13065v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2208.13065](http://arxiv.org/abs/2208.13065)

    本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。

    

    通常，系统操作员在开环预测优化过程中进行电力系统的经济运行：首先预测可再生能源(RES)的可用性和系统储备需求；根据这些预测，系统操作员解决诸如机组组合(UC)的优化模型，以确定相应的经济运行计划。然而，这种开环过程可能会实质性地损害操作经济性，因为它的预测器目光短浅地寻求改善即时的统计预测误差，而不是最终的操作成本。为此，本文提出了一个闭环预测优化框架，提供一种预测机组组合以改善操作经济性的方法。首先，利用双层混合整数规划模型针对最佳系统操作训练成本导向的预测器。上层基于其引起的操作成本来训练 RES 和储备预测器；下层则在给定预测的 RES 和储备的情况下，依据最佳操作原则求解 UC。这两个层级通过反馈环路进行交互性互动，直到收敛为止。在修改后的IEEE 24-bus系统上的数值实验表明，与三种最先进的 UC 基准线相比，所提出的框架具有高效性和有效性。

    Generally, system operators conduct the economic operation of power systems in an open-loop predict-then-optimize process: the renewable energy source (RES) availability and system reserve requirements are first predicted; given the predictions, system operators solve optimization models such as unit commitment (UC) to determine the economical operation plans accordingly. However, such an open-loop process could essentially compromise the operation economics because its predictors myopically seek to improve the immediate statistical prediction errors instead of the ultimate operation cost. To this end, this paper presents a closed-loop predict-and-optimize framework, offering a prescriptive UC to improve the operation economics. First, a bilevel mixed-integer programming model is leveraged to train cost-oriented predictors tailored for optimal system operations: the upper level trains the RES and reserve predictors based on their induced operation cost; the lower level, with given pred
    
[^5]: 多个弱工具的二进制响应模型

    Binary response model with many weak instruments. (arXiv:2201.04811v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2201.04811](http://arxiv.org/abs/2201.04811)

    本文提供了一种使用控制函数方法和正则化方案来获得更好内生二进制响应模型估计结果的方法，并应用于研究家庭收入对大学完成的影响。

    

    本文考虑了具有许多弱工具的内生二进制响应模型。我们采用控制函数方法和正则化方案，在存在许多弱工具的情况下获得更好的内生二进制响应模型估计结果。提供了两个一致且渐近正态分布的估计器，分别称为正则化条件最大似然估计器（RCMLE）和正则化非线性最小二乘估计器（RNLSE）。Monte Carlo模拟表明，所提出的估计量在存在许多弱工具时优于现有的估计量。我们应用估计方法研究家庭收入对大学完成的影响。

    This paper considers an endogenous binary response model with many weak instruments. We in the current paper employ a control function approach and a regularization scheme to obtain better estimation results for the endogenous binary response model in the presence of many weak instruments. Two consistent and asymptotically normally distributed estimators are provided, each of which is called a regularized conditional maximum likelihood estimator (RCMLE) and a regularized nonlinear least square estimator (RNLSE) respectively. Monte Carlo simulations show that the proposed estimators outperform the existing estimators when many weak instruments are present. We apply our estimation method to study the effect of family income on college completion.
    

