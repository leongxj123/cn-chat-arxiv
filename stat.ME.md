# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal thinking for decision making on Electronic Health Records: why and how.](http://arxiv.org/abs/2308.01605) | 本文介绍了在电子健康记录中使用因果思维进行决策的必要性和方法。通过模拟随机试验来个性化决策，以减少数据中的偏见。这对于分析电子健康记录或索赔数据以得出因果结论的最重要陷阱和考虑因素进行了重点强调。 |
| [^2] | [Robust Estimation and Inference in Panels with Interactive Fixed Effects.](http://arxiv.org/abs/2210.06639) | 本文研究了具有交互固定效应的面板数据中回归系数的估计和推断问题。通过采用改进的估计器和偏倚感知置信区间，我们能够解决因素弱引起的偏倚和大小失真的问题，无论因素是否强壮，都能得到统一有效的结果。 |

# 详细

[^1]: 用于决策的因果思维在电子健康记录中的应用：为什么以及如何

    Causal thinking for decision making on Electronic Health Records: why and how. (arXiv:2308.01605v1 [stat.ME])

    [http://arxiv.org/abs/2308.01605](http://arxiv.org/abs/2308.01605)

    本文介绍了在电子健康记录中使用因果思维进行决策的必要性和方法。通过模拟随机试验来个性化决策，以减少数据中的偏见。这对于分析电子健康记录或索赔数据以得出因果结论的最重要陷阱和考虑因素进行了重点强调。

    

    准确的预测，如同机器学习一样，可能无法为每个患者提供最佳医疗保健。确实，预测可能受到数据中的捷径（如种族偏见）的驱动。为数据驱动的决策需要因果思维。在这里，我们介绍关键要素，重点关注常规收集的数据，即电子健康记录（EHRs）和索赔数据。使用这些数据评估干预的价值需要谨慎：时间依赖性和现有实践很容易混淆因果效应。我们提供了一个逐步框架，帮助从真实患者记录中构建有效的决策，通过模拟随机试验来个性化决策，例如使用机器学习。我们的框架强调了分析EHRs或索赔数据以得出因果结论时最重要的陷阱和考虑因素。我们在用于重症医学信息市场中的肌酐对败血症死亡率的影响的研究中说明了各种选择。

    Accurate predictions, as with machine learning, may not suffice to provide optimal healthcare for every patient. Indeed, prediction can be driven by shortcuts in the data, such as racial biases. Causal thinking is needed for data-driven decisions. Here, we give an introduction to the key elements, focusing on routinely-collected data, electronic health records (EHRs) and claims data. Using such data to assess the value of an intervention requires care: temporal dependencies and existing practices easily confound the causal effect. We present a step-by-step framework to help build valid decision making from real-life patient records by emulating a randomized trial before individualizing decisions, eg with machine learning. Our framework highlights the most important pitfalls and considerations in analysing EHRs or claims data to draw causal conclusions. We illustrate the various choices in studying the effect of albumin on sepsis mortality in the Medical Information Mart for Intensive C
    
[^2]: 具有交互固定效应的面板数据中的鲁棒估计和推断

    Robust Estimation and Inference in Panels with Interactive Fixed Effects. (arXiv:2210.06639v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.06639](http://arxiv.org/abs/2210.06639)

    本文研究了具有交互固定效应的面板数据中回归系数的估计和推断问题。通过采用改进的估计器和偏倚感知置信区间，我们能够解决因素弱引起的偏倚和大小失真的问题，无论因素是否强壮，都能得到统一有效的结果。

    

    本文考虑具有交互固定效应（即具有因子结构）的面板数据中回归系数的估计和推断问题。我们发现之前开发的估计器和置信区间可能在一些因素较弱的情况下存在严重的偏倚和大小失真。我们提出了具有改进收敛速度和偏倚感知置信区间的估计器，无论因素是否强壮都能保持统一有效。我们的方法采用最小化线性估计理论，在初始交互固定效应的误差上使用核范数约束来形成一个无偏估计。我们利用所得估计构建一个考虑到因素弱引起的剩余偏差的偏倚感知置信区间。在蒙特卡洛实验中，我们发现在因素较弱的情况下相较于传统方法有显著改进，并且在因素较强的情况下几乎没有估计误差的损失。

    We consider estimation and inference for a regression coefficient in panels with interactive fixed effects (i.e., with a factor structure). We show that previously developed estimators and confidence intervals (CIs) might be heavily biased and size-distorted when some of the factors are weak. We propose estimators with improved rates of convergence and bias-aware CIs that are uniformly valid regardless of whether the factors are strong or not. Our approach applies the theory of minimax linear estimation to form a debiased estimate using a nuclear norm bound on the error of an initial estimate of the interactive fixed effects. We use the obtained estimate to construct a bias-aware CI taking into account the remaining bias due to weak factors. In Monte Carlo experiments, we find a substantial improvement over conventional approaches when factors are weak, with little cost to estimation error when factors are strong.
    

