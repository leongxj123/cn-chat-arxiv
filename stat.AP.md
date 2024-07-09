# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Forecasting Volatility of Oil-based Commodities: The Model of Dynamic Persistence](https://rss.arxiv.org/abs/2402.01354) | 该论文提出了一种新的方法来预测基于石油的商品的波动性，通过将异质持续性的冲击在时间上平滑变化，这种模型在改进波动性预测方面表现出色，并且特别适用于较长时间段的预测。 |
| [^2] | [Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment.](http://arxiv.org/abs/2208.13065) | 本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。 |

# 详细

[^1]: 预测基于石油的商品波动性: 动态持续性模型

    Forecasting Volatility of Oil-based Commodities: The Model of Dynamic Persistence

    [https://rss.arxiv.org/abs/2402.01354](https://rss.arxiv.org/abs/2402.01354)

    该论文提出了一种新的方法来预测基于石油的商品的波动性，通过将异质持续性的冲击在时间上平滑变化，这种模型在改进波动性预测方面表现出色，并且特别适用于较长时间段的预测。

    

    时间变动和持续性是波动性的关键属性，通常在基于石油的波动性预测模型中分别研究。在这里，我们提出了一种新的方法，允许具有异质持续性的冲击在时间上平滑变化，并将两者结合在一起建模。我们认为这很重要，因为这种动态是由于基于石油商品的冲击的动态性质自然产生的。我们通过局部回归从数据中识别出这种动态，并建立了一个显著改进波动性预测的模型。这种基于在时间上平滑变化的丰富持续性结构的预测模型，超越了最先进的基准模型，并在较长的预测时间段内特别有用。

    Time variation and persistence are crucial properties of volatility that are often studied separately in oil-based volatility forecasting models. Here, we propose a novel approach that allows shocks with heterogeneous persistence to vary smoothly over time, and thus model the two together. We argue that this is important because such dynamics arise naturally from the dynamic nature of shocks in oil-based commodities. We identify such dynamics from the data using localised regressions and build a model that significantly improves volatility forecasts. Such forecasting models, based on a rich persistence structure that varies smoothly over time, outperform state-of-the-art benchmark models and are particularly useful for forecasting over longer horizons.
    
[^2]: 改善运营经济学：基于双层 MIP 的闭环预测优化框架来预测机组组合的操作计划

    Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment. (arXiv:2208.13065v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2208.13065](http://arxiv.org/abs/2208.13065)

    本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。

    

    通常，系统操作员在开环预测优化过程中进行电力系统的经济运行：首先预测可再生能源(RES)的可用性和系统储备需求；根据这些预测，系统操作员解决诸如机组组合(UC)的优化模型，以确定相应的经济运行计划。然而，这种开环过程可能会实质性地损害操作经济性，因为它的预测器目光短浅地寻求改善即时的统计预测误差，而不是最终的操作成本。为此，本文提出了一个闭环预测优化框架，提供一种预测机组组合以改善操作经济性的方法。首先，利用双层混合整数规划模型针对最佳系统操作训练成本导向的预测器。上层基于其引起的操作成本来训练 RES 和储备预测器；下层则在给定预测的 RES 和储备的情况下，依据最佳操作原则求解 UC。这两个层级通过反馈环路进行交互性互动，直到收敛为止。在修改后的IEEE 24-bus系统上的数值实验表明，与三种最先进的 UC 基准线相比，所提出的框架具有高效性和有效性。

    Generally, system operators conduct the economic operation of power systems in an open-loop predict-then-optimize process: the renewable energy source (RES) availability and system reserve requirements are first predicted; given the predictions, system operators solve optimization models such as unit commitment (UC) to determine the economical operation plans accordingly. However, such an open-loop process could essentially compromise the operation economics because its predictors myopically seek to improve the immediate statistical prediction errors instead of the ultimate operation cost. To this end, this paper presents a closed-loop predict-and-optimize framework, offering a prescriptive UC to improve the operation economics. First, a bilevel mixed-integer programming model is leveraged to train cost-oriented predictors tailored for optimal system operations: the upper level trains the RES and reserve predictors based on their induced operation cost; the lower level, with given pred
    

