# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Larger the Better? Improved LLM Code-Generation via Budget Reallocation](https://arxiv.org/abs/2404.00725) | 较小的语言模型可以在相同预算下产生可靠的改进，但在无法进行单元测试的情况下，较小的模型选择排名次于较大模型的单个输出。 |
| [^2] | [Brand Network Booster: A New System for Improving Brand Connectivity.](http://arxiv.org/abs/2309.16228) | 本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。 |

# 详细

[^1]: 越大越好吗？通过预算重新分配改进LLM代码生成

    The Larger the Better? Improved LLM Code-Generation via Budget Reallocation

    [https://arxiv.org/abs/2404.00725](https://arxiv.org/abs/2404.00725)

    较小的语言模型可以在相同预算下产生可靠的改进，但在无法进行单元测试的情况下，较小的模型选择排名次于较大模型的单个输出。

    

    人们普遍认为，大型语言模型(LLMs)比较小的模型更好。然而，更大的模型在推断过程中也需要更多的时间和计算资源。这就引出了一个问题：当两个模型在相同的预算下运行时会发生什么？（例如，计算资源，运行时间）。为了解决这个问题，我们分析了各种大小的代码生成LLMs，并进行比较，例如运行一个70B模型一次与从13B模型生成五个输出并选择一个的情况。我们的研究结果表明，在标准单元测试设置中，反复使用较小的模型可以产生一致的改进，在五个任务中最高可达15%的增益。另一方面，在无法进行单元测试的情况下，从较小模型中基于排名的候选选择表现不及来自较大模型的单个输出。我们的结果突显了使用较小模型而非较大模型的潜力。

    arXiv:2404.00725v1 Announce Type: cross  Abstract: It is a common belief that large language models (LLMs) are better than smaller-sized ones. However, larger models also require significantly more time and compute during inference. This begs the question: what happens when both models operate under the same budget? (e.g., compute, run-time). To address this question, we analyze code generation LLMs of various sizes and make comparisons such as running a 70B model once vs. generating five outputs from a 13B model and selecting one. Our findings reveal that, in a standard unit-test setup, the repeated use of smaller models can yield consistent improvements, with gains of up to 15% across five tasks. On the other hand, in scenarios where unit-tests are unavailable, a ranking-based selection of candidates from the smaller model falls short of the performance of a single output from larger ones. Our results highlight the potential of using smaller models instead of larger ones, and the imp
    
[^2]: 品牌网络增强器：提升品牌连接性的新系统

    Brand Network Booster: A New System for Improving Brand Connectivity. (arXiv:2309.16228v1 [cs.SI])

    [http://arxiv.org/abs/2309.16228](http://arxiv.org/abs/2309.16228)

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。

    

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。在网络分析方面，我们通过解决扩展版的最大连接度改进问题来实现这一目标，其中包括考虑敌对节点、约束预算和加权网络的可能性 - 通过添加链接或增加现有连接的权重来实现连接性的改进。我们结合两个案例研究来展示这个新系统，并讨论其性能。我们的工具和方法对于网络学者和支持市场营销和传播管理者的战略决策过程都很有用。

    This paper presents a new decision support system offered for an in-depth analysis of semantic networks, which can provide insights for a better exploration of a brand's image and the improvement of its connectivity. In terms of network analysis, we show that this goal is achieved by solving an extended version of the Maximum Betweenness Improvement problem, which includes the possibility of considering adversarial nodes, constrained budgets, and weighted networks - where connectivity improvement can be obtained by adding links or increasing the weight of existing connections. We present this new system together with two case studies, also discussing its performance. Our tool and approach are useful both for network scholars and for supporting the strategic decision-making processes of marketing and communication managers.
    

