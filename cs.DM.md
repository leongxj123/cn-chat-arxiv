# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Plum: Prompt Learning using Metaheuristic](https://arxiv.org/abs/2311.08364) | 提出了使用元启发式的提示学习方法，通过测试六种典型的元启发式方法，在大型语言模型的提示优化任务中取得了有效性。 |
| [^2] | [Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs.](http://arxiv.org/abs/2401.13054) | 本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。 |

# 详细

[^1]: Plum: 使用元启发式的提示学习

    Plum: Prompt Learning using Metaheuristic

    [https://arxiv.org/abs/2311.08364](https://arxiv.org/abs/2311.08364)

    提出了使用元启发式的提示学习方法，通过测试六种典型的元启发式方法，在大型语言模型的提示优化任务中取得了有效性。

    

    自从大型语言模型出现以来，提示学习已成为优化和定制这些模型的一种流行方法。特殊提示，如“思维链”，甚至揭示了这些模型内部先前未知的推理能力。然而，发现有效提示的进展缓慢，促使人们渴望一种通用的提示优化方法。不幸的是，现有的提示学习方法中很少有满足“通用”的标准，即同时具备自动、离散、黑盒、无梯度和可解释性。在本文中，我们引入元启发式，作为一种有希望的提示学习方法的离散非凸优化方法分支，拥有100多种选项。在我们的范式中，我们测试了六种典型方法：爬山、模拟退火、遗传算法（带/不带交叉）、禁忌搜索和和谐搜索，展示了它们在白盒模式下的有效性。

    arXiv:2311.08364v2 Announce Type: replace-cross  Abstract: Since the emergence of large language models, prompt learning has become a popular method for optimizing and customizing these models. Special prompts, such as Chain-of-Thought, have even revealed previously unknown reasoning capabilities within these models. However, the progress of discovering effective prompts has been slow, driving a desire for general prompt optimization methods. Unfortunately, few existing prompt learning methods satisfy the criteria of being truly "general", i.e., automatic, discrete, black-box, gradient-free, and interpretable all at once. In this paper, we introduce metaheuristics, a branch of discrete non-convex optimization methods with over 100 options, as a promising approach to prompt learning. Within our paradigm, we test six typical methods: hill climbing, simulated annealing, genetic algorithms with/without crossover, tabu search, and harmony search, demonstrating their effectiveness in white-b
    
[^2]: 无计算困难的快速计算超图节点距离的方法

    Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs. (arXiv:2401.13054v1 [cs.SI])

    [http://arxiv.org/abs/2401.13054](http://arxiv.org/abs/2401.13054)

    本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。

    

    超图是图的推广，当考虑实体间的属性共享时会自然产生。尽管可以通过将超边扩展为完全连接的子图来将超图转换为图，但逆向操作在计算上非常复杂且属于NP-complete问题。因此，我们假设超图包含比图更多的信息。此外，直接操作超图比将其扩展为图更为方便。超图中的一个开放问题是如何精确高效地计算节点之间的距离。通过估计节点距离，我们能够找到节点的最近邻居，并使用K最近邻（KNN）方法在超图上执行标签传播。在本文中，我们提出了一种基于随机游走的新方法，实现了在超图上进行标签传播。我们将节点距离估计为随机游走的预期到达时间。我们注意到简单随机游走（SRW）无法准确描述节点之间的距离，因此我们引入了"frustrated"的概念。

    A hypergraph is a generalization of a graph that arises naturally when attribute-sharing among entities is considered. Although a hypergraph can be converted into a graph by expanding its hyperedges into fully connected subgraphs, going the reverse way is computationally complex and NP-complete. We therefore hypothesize that a hypergraph contains more information than a graph. In addition, it is more convenient to manipulate a hypergraph directly, rather than expand it into a graph. An open problem in hypergraphs is how to accurately and efficiently calculate their node distances. Estimating node distances enables us to find a node's nearest neighbors, and perform label propagation on hypergraphs using a K-nearest neighbors (KNN) approach. In this paper, we propose a novel approach based on random walks to achieve label propagation on hypergraphs. We estimate node distances as the expected hitting times of random walks. We note that simple random walks (SRW) cannot accurately describe 
    

