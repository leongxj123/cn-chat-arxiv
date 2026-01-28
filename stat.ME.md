# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inflation Target at Risk: A Time-varying Parameter Distributional Regression](https://arxiv.org/abs/2403.12456) | 介绍了一种新颖的半参数方法，用于构建时间变化的条件分布，通过分布回归估计所有模型参数，预测美国通货膨胀分布的风险。 |
| [^2] | [Statistical Hypothesis Testing for Information Value (IV).](http://arxiv.org/abs/2309.13183) | 该论文提出了信息价值（IV）的统计假设检验方法，为模型建立前的特征选择提供了理论框架，并通过实验证明了该方法的有效性。 |

# 详细

[^1]: 通货膨胀目标风险：一种时变参数分布回归

    Inflation Target at Risk: A Time-varying Parameter Distributional Regression

    [https://arxiv.org/abs/2403.12456](https://arxiv.org/abs/2403.12456)

    介绍了一种新颖的半参数方法，用于构建时间变化的条件分布，通过分布回归估计所有模型参数，预测美国通货膨胀分布的风险。

    

    宏观变量经常显示时间变化的分布，这是由经济、社会和环境因素的动态和演变特征所驱动的，这些因素持续地重塑着统治这些变量的基本模式和关系。为了更好地理解超出中心趋势的分布动态，本文引入了一种新颖的半参数方法，用于构建时间变化的条件分布，依赖于分布回归的最新进展。我们提出了一种高效的基于精度的马尔可夫链蒙特卡罗算法，可以同时估计所有模型参数，同时明确地施加条件分布函数上的单调性条件。我们的模型被应用于构建美国通货膨胀的预测分布，条件于一组宏观经济和金融指标。未来通货膨胀偏离过高或过低的风险

    arXiv:2403.12456v1 Announce Type: new  Abstract: Macro variables frequently display time-varying distributions, driven by the dynamic and evolving characteristics of economic, social, and environmental factors that consistently reshape the fundamental patterns and relationships governing these variables. To better understand the distributional dynamics beyond the central tendency, this paper introduces a novel semi-parametric approach for constructing time-varying conditional distributions, relying on the recent advances in distributional regression. We present an efficient precision-based Markov Chain Monte Carlo algorithm that simultaneously estimates all model parameters while explicitly enforcing the monotonicity condition on the conditional distribution function. Our model is applied to construct the forecasting distribution of inflation for the U.S., conditional on a set of macroeconomic and financial indicators. The risks of future inflation deviating excessively high or low fro
    
[^2]: 信息价值（IV）的统计假设检验

    Statistical Hypothesis Testing for Information Value (IV). (arXiv:2309.13183v1 [math.ST])

    [http://arxiv.org/abs/2309.13183](http://arxiv.org/abs/2309.13183)

    该论文提出了信息价值（IV）的统计假设检验方法，为模型建立前的特征选择提供了理论框架，并通过实验证明了该方法的有效性。

    

    信息价值（IV）是模型建立前进行特征选择的一种常用技术。目前存在一些实际标准，但基于IV的判断是否一个预测因子具有足够的预测能力的理论依据依然神秘且缺乏。然而，关于该技术的数学发展和统计推断方法在文献中几乎没有提及。在本研究中，我们提出了一个关于IV的理论框架，并提出了一种非参数假设检验方法来测试预测能力。我们展示了如何高效计算检验统计量，并在模拟数据上研究其表现。此外，我们将这一方法应用于银行欺诈数据，并提供了一个实现我们结果的Python库。

    Information value (IV) is a quite popular technique for feature selection prior to the modeling phase. There are practical criteria, but at the same time mysterious and lacking theoretical arguments, based on the IV, to decide if a predictor has sufficient predictive power to be considered in the modeling phase. However, the mathematical development and statistical inference methods for this technique is almost non-existent in the literature. In this work we present a theoretical framework for the IV and propose a non-parametric hypothesis test to test the predictive power. We show how to efficiently calculate the test statistic and study its performance on simulated data. Additionally, we apply our test on bank fraud data and provide a Python library where we implement our results.
    

