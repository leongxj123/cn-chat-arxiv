# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-View Symbolic Regression](https://arxiv.org/abs/2402.04298) | 多视角符号回归(MvSR)是一种同时考虑多个数据集的符号回归方法，能够找到一个参数化解来准确拟合所有数据集，解决了传统方法无法处理不同实验设置的问题。 |

# 详细

[^1]: 多视角符号回归

    Multi-View Symbolic Regression

    [https://arxiv.org/abs/2402.04298](https://arxiv.org/abs/2402.04298)

    多视角符号回归(MvSR)是一种同时考虑多个数据集的符号回归方法，能够找到一个参数化解来准确拟合所有数据集，解决了传统方法无法处理不同实验设置的问题。

    

    符号回归(SR)搜索表示解释变量和响应变量之间关系的分析表达式。目前的SR方法假设从单个实验中提取的单个数据集。然而，研究人员经常面临来自不同设置的多个实验结果集。传统的SR方法可能无法找到潜在的表达式，因为每个实验的参数可能不同。在这项工作中，我们提出了多视角符号回归(MvSR)，它同时考虑多个数据集，模拟实验环境，并输出一个通用的参数化解。这种方法将评估的表达式适应每个独立数据集，并同时返回能够准确拟合所有数据集的参数函数族f(x; \theta)。我们使用从已知表达式生成的数据以及来自实际世界的数据来展示MvSR的有效性。

    Symbolic regression (SR) searches for analytical expressions representing the relationship between a set of explanatory and response variables. Current SR methods assume a single dataset extracted from a single experiment. Nevertheless, frequently, the researcher is confronted with multiple sets of results obtained from experiments conducted with different setups. Traditional SR methods may fail to find the underlying expression since the parameters of each experiment can be different. In this work we present Multi-View Symbolic Regression (MvSR), which takes into account multiple datasets simultaneously, mimicking experimental environments, and outputs a general parametric solution. This approach fits the evaluated expression to each independent dataset and returns a parametric family of functions f(x; \theta) simultaneously capable of accurately fitting all datasets. We demonstrate the effectiveness of MvSR using data generated from known expressions, as well as real-world data from 
    

