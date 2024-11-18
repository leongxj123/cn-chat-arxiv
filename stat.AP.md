# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-View Symbolic Regression](https://arxiv.org/abs/2402.04298) | 多视角符号回归(MvSR)是一种同时考虑多个数据集的符号回归方法，能够找到一个参数化解来准确拟合所有数据集，解决了传统方法无法处理不同实验设置的问题。 |
| [^2] | [Time-Varying Parameters as Ridge Regressions.](http://arxiv.org/abs/2009.00401) | 该论文提出了一种实际上是基于岭回归的时变参数模型，这比传统的状态空间方法计算更快，调整更容易，有助于研究经济结构性变化。 |

# 详细

[^1]: 多视角符号回归

    Multi-View Symbolic Regression

    [https://arxiv.org/abs/2402.04298](https://arxiv.org/abs/2402.04298)

    多视角符号回归(MvSR)是一种同时考虑多个数据集的符号回归方法，能够找到一个参数化解来准确拟合所有数据集，解决了传统方法无法处理不同实验设置的问题。

    

    符号回归(SR)搜索表示解释变量和响应变量之间关系的分析表达式。目前的SR方法假设从单个实验中提取的单个数据集。然而，研究人员经常面临来自不同设置的多个实验结果集。传统的SR方法可能无法找到潜在的表达式，因为每个实验的参数可能不同。在这项工作中，我们提出了多视角符号回归(MvSR)，它同时考虑多个数据集，模拟实验环境，并输出一个通用的参数化解。这种方法将评估的表达式适应每个独立数据集，并同时返回能够准确拟合所有数据集的参数函数族f(x; \theta)。我们使用从已知表达式生成的数据以及来自实际世界的数据来展示MvSR的有效性。

    Symbolic regression (SR) searches for analytical expressions representing the relationship between a set of explanatory and response variables. Current SR methods assume a single dataset extracted from a single experiment. Nevertheless, frequently, the researcher is confronted with multiple sets of results obtained from experiments conducted with different setups. Traditional SR methods may fail to find the underlying expression since the parameters of each experiment can be different. In this work we present Multi-View Symbolic Regression (MvSR), which takes into account multiple datasets simultaneously, mimicking experimental environments, and outputs a general parametric solution. This approach fits the evaluated expression to each independent dataset and returns a parametric family of functions f(x; \theta) simultaneously capable of accurately fitting all datasets. We demonstrate the effectiveness of MvSR using data generated from known expressions, as well as real-world data from 
    
[^2]: 使用岭回归法的时变参数模型

    Time-Varying Parameters as Ridge Regressions. (arXiv:2009.00401v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2009.00401](http://arxiv.org/abs/2009.00401)

    该论文提出了一种实际上是基于岭回归的时变参数模型，这比传统的状态空间方法计算更快，调整更容易，有助于研究经济结构性变化。

    

    时变参数模型(TVPs)经常被用于经济学中来捕捉结构性变化。我强调了一个被忽视的事实——这些实际上是岭回归。这使得计算、调整和实现比状态空间范式更容易。在高维情况下，解决等价的双重岭问题的计算非常快,关键的“时间变化量”通常是由交叉验证来调整的。使用两步回归岭回归来处理不断变化的波动性。我考虑了基于稀疏性(算法选择哪些参数变化, 哪些不变)和降低秩约束的扩展(变化与因子模型相关联)。为了展示这种方法的有用性, 我使用它来研究加拿大货币政策的演变, 并使用大规模时变局部投影估计约4600个TVPs, 这一任务完全可以利用这种新方法完成。

    Time-varying parameters (TVPs) models are frequently used in economics to capture structural change. I highlight a rather underutilized fact -- that these are actually ridge regressions. Instantly, this makes computations, tuning, and implementation much easier than in the state-space paradigm. Among other things, solving the equivalent dual ridge problem is computationally very fast even in high dimensions, and the crucial "amount of time variation" is tuned by cross-validation. Evolving volatility is dealt with using a two-step ridge regression. I consider extensions that incorporate sparsity (the algorithm selects which parameters vary and which do not) and reduced-rank restrictions (variation is tied to a factor model). To demonstrate the usefulness of the approach, I use it to study the evolution of monetary policy in Canada using large time-varying local projections. The application requires the estimation of about 4600 TVPs, a task well within the reach of the new method.
    

