# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Holistic Indicator of Polarization to Measure Online Sexism](https://arxiv.org/abs/2404.02205) | 该论文提出了一个可以提供综合性性别毒性指标的模型，有助于政策制定者、在线社区管理员和计算社会科学家更好地理解和管理在线性别歧视问题。 |
| [^2] | [Novel Node Category Detection Under Subpopulation Shift](https://arxiv.org/abs/2404.01216) | 提出了一种新方法 RECO-SLIP，用于在属性图中检测属于新类别的节点，能够有效解决子群体转移下的节点检测问题，实验证明其性能优越。 |
| [^3] | [A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation](https://arxiv.org/abs/2402.03358) | 这篇综述调研了图缩减方法，包括稀疏化、粗化和浓缩，在解决大型图形数据分析和计算复杂性方面起到了重要作用。调研对这些方法的技术细节进行了系统的回顾，并强调了它们在实际应用中的关键性。同时，调研还提出了保证图缩减技术持续有效性的关键研究方向。 |
| [^4] | [Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs.](http://arxiv.org/abs/2401.13054) | 本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。 |
| [^5] | [A Two-Part Machine Learning Approach to Characterizing Network Interference in A/B Testing.](http://arxiv.org/abs/2308.09790) | 本论文提出了一种两部分机器学习方法，用于识别和描述A/B测试中的网络干扰。通过考虑潜在的复杂网络结构和建立适合的曝光映射，该方法在合成实验和真实大规模测试中的模拟中表现优于传统方法。 |
| [^6] | [On the convergence of nonlinear averaging dynamics with three-body interactions on hypergraphs.](http://arxiv.org/abs/2304.07203) | 本文研究了超图上具有三体相互作用的离散时间非线性平均动力学，在初态和超图拓扑以及更新非线性相互作用下，产生了高阶动力效应。 |

# 详细

[^1]: 一个整体性极化指标以衡量在线性别歧视

    A Holistic Indicator of Polarization to Measure Online Sexism

    [https://arxiv.org/abs/2404.02205](https://arxiv.org/abs/2404.02205)

    该论文提出了一个可以提供综合性性别毒性指标的模型，有助于政策制定者、在线社区管理员和计算社会科学家更好地理解和管理在线性别歧视问题。

    

    arXiv:2404.02205v1 公告类型: 跨领域 摘要: 男权主义者和女权主义者在社交网络上的在线趋势需要一个整体性的衡量性别歧视水平的指标。这个指标对于政策制定者和在线社区的管理员（如subreddits）以及计算社会科学家至关重要，他们可以根据性别歧视程度修改管理策略，或者匹配和比较不同平台和社区上的时态性别歧视，以及推断社会科学见解。在本文中，我们建立了一个模型，可以提供一个可比较的针对男性、女性身份以及男性、女性个人的毒性整体指标。尽管先前的监督NLP方法需要在目标级别对有毒评论进行注释（例如注释特别针对女性有毒的评论）来检测针对性有毒评论，我们的指标利用监督式NLP来检测毒性的存在。

    arXiv:2404.02205v1 Announce Type: cross  Abstract: The online trend of the manosphere and feminist discourse on social networks requires a holistic measure of the level of sexism in an online community. This indicator is important for policymakers and moderators of online communities (e.g., subreddits) and computational social scientists, either to revise moderation strategies based on the degree of sexism or to match and compare the temporal sexism across different platforms and communities with real-time events and infer social scientific insights.   In this paper, we build a model that can provide a comparable holistic indicator of toxicity targeted toward male and female identity and male and female individuals. Despite previous supervised NLP methods that require annotation of toxic comments at the target level (e.g. annotating comments that are specifically toxic toward women) to detect targeted toxic comments, our indicator uses supervised NLP to detect the presence of toxicity 
    
[^2]: 在子群体转移下的新颖节点类别检测

    Novel Node Category Detection Under Subpopulation Shift

    [https://arxiv.org/abs/2404.01216](https://arxiv.org/abs/2404.01216)

    提出了一种新方法 RECO-SLIP，用于在属性图中检测属于新类别的节点，能够有效解决子群体转移下的节点检测问题，实验证明其性能优越。

    

    在现实世界的图数据中，分布转移可以通过各种方式表现，例如新类别的出现和现有类别相对比例的变化。在这种分布转移下，检测属于新类别的节点对于安全或洞察发现至关重要。我们引入了一种新方法，称为具有选择性链路预测的召回约束优化（RECO-SLIP），用于在子群体转移下检测属性图中属于新类别的节点。通过将召回约束学习框架与高效样本预测机制相结合，RECO-SLIP解决了抵抗子群体转移和有效利用图结构的双重挑战。我们在多个图数据集上进行了大量实证评估，结果表明RECO-SLIP相对于现有方法具有更优异的性能。

    arXiv:2404.01216v1 Announce Type: new  Abstract: In real-world graph data, distribution shifts can manifest in various ways, such as the emergence of new categories and changes in the relative proportions of existing categories. It is often important to detect nodes of novel categories under such distribution shifts for safety or insight discovery purposes. We introduce a new approach, Recall-Constrained Optimization with Selective Link Prediction (RECO-SLIP), to detect nodes belonging to novel categories in attributed graphs under subpopulation shifts. By integrating a recall-constrained learning framework with a sample-efficient link prediction mechanism, RECO-SLIP addresses the dual challenges of resilience against subpopulation shifts and the effective exploitation of graph structure. Our extensive empirical evaluation across multiple graph datasets demonstrates the superior performance of RECO-SLIP over existing methods.
    
[^3]: 图缩减的综合调研：稀疏化、粗化和浓缩

    A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation

    [https://arxiv.org/abs/2402.03358](https://arxiv.org/abs/2402.03358)

    这篇综述调研了图缩减方法，包括稀疏化、粗化和浓缩，在解决大型图形数据分析和计算复杂性方面起到了重要作用。调研对这些方法的技术细节进行了系统的回顾，并强调了它们在实际应用中的关键性。同时，调研还提出了保证图缩减技术持续有效性的关键研究方向。

    

    许多真实世界的数据集可以自然地表示为图，涵盖了广泛的领域。然而，图数据集的复杂性和规模的增加为分析和计算带来了显著的挑战。为此，图缩减技术在保留关键属性的同时简化大型图形数据变得越来越受关注。在本调研中，我们旨在提供对图缩减方法的全面理解，包括图稀疏化、图粗化和图浓缩。具体而言，我们建立了这些方法的统一定义，并引入了一个分层分类法来分类这些方法所解决的挑战。我们的调研系统地回顾了这些方法的技术细节，并强调了它们在各种场景中的实际应用。此外，我们还概述了保证图缩减技术持续有效性的关键研究方向，并提供了一个详细的论文列表链接。

    Many real-world datasets can be naturally represented as graphs, spanning a wide range of domains. However, the increasing complexity and size of graph datasets present significant challenges for analysis and computation. In response, graph reduction techniques have gained prominence for simplifying large graphs while preserving essential properties. In this survey, we aim to provide a comprehensive understanding of graph reduction methods, including graph sparsification, graph coarsening, and graph condensation. Specifically, we establish a unified definition for these methods and introduce a hierarchical taxonomy to categorize the challenges they address. Our survey then systematically reviews the technical details of these methods and emphasizes their practical applications across diverse scenarios. Furthermore, we outline critical research directions to ensure the continued effectiveness of graph reduction techniques, as well as provide a comprehensive paper list at https://github.
    
[^4]: 无计算困难的快速计算超图节点距离的方法

    Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs. (arXiv:2401.13054v1 [cs.SI])

    [http://arxiv.org/abs/2401.13054](http://arxiv.org/abs/2401.13054)

    本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。

    

    超图是图的推广，当考虑实体间的属性共享时会自然产生。尽管可以通过将超边扩展为完全连接的子图来将超图转换为图，但逆向操作在计算上非常复杂且属于NP-complete问题。因此，我们假设超图包含比图更多的信息。此外，直接操作超图比将其扩展为图更为方便。超图中的一个开放问题是如何精确高效地计算节点之间的距离。通过估计节点距离，我们能够找到节点的最近邻居，并使用K最近邻（KNN）方法在超图上执行标签传播。在本文中，我们提出了一种基于随机游走的新方法，实现了在超图上进行标签传播。我们将节点距离估计为随机游走的预期到达时间。我们注意到简单随机游走（SRW）无法准确描述节点之间的距离，因此我们引入了"frustrated"的概念。

    A hypergraph is a generalization of a graph that arises naturally when attribute-sharing among entities is considered. Although a hypergraph can be converted into a graph by expanding its hyperedges into fully connected subgraphs, going the reverse way is computationally complex and NP-complete. We therefore hypothesize that a hypergraph contains more information than a graph. In addition, it is more convenient to manipulate a hypergraph directly, rather than expand it into a graph. An open problem in hypergraphs is how to accurately and efficiently calculate their node distances. Estimating node distances enables us to find a node's nearest neighbors, and perform label propagation on hypergraphs using a K-nearest neighbors (KNN) approach. In this paper, we propose a novel approach based on random walks to achieve label propagation on hypergraphs. We estimate node distances as the expected hitting times of random walks. We note that simple random walks (SRW) cannot accurately describe 
    
[^5]: 一种用于描述A/B测试中网络干扰的两部分机器学习方法

    A Two-Part Machine Learning Approach to Characterizing Network Interference in A/B Testing. (arXiv:2308.09790v1 [stat.ML])

    [http://arxiv.org/abs/2308.09790](http://arxiv.org/abs/2308.09790)

    本论文提出了一种两部分机器学习方法，用于识别和描述A/B测试中的网络干扰。通过考虑潜在的复杂网络结构和建立适合的曝光映射，该方法在合成实验和真实大规模测试中的模拟中表现优于传统方法。

    

    受网络干扰现象的影响，控制实验或"A/B测试"的可靠性通常会受到损害。为了解决这个问题，我们提出了一种基于机器学习的方法来识别和描述异质网络干扰。我们的方法考虑了潜在的复杂网络结构，并自动化了"曝光映射"确定的任务，从而解决了现有文献中的两个主要限制。我们引入了"因果网络模式"，并采用透明的机器学习模型来建立最适合反映潜在网络干扰模式的曝光映射。我们的方法通过在两个合成实验和一个涉及100-200万Instagram用户的真实大规模测试中的模拟得到了验证，表现优于传统方法，如基于设计的集群随机化和基于分析的邻域曝光映射。

    The reliability of controlled experiments, or "A/B tests," can often be compromised due to the phenomenon of network interference, wherein the outcome for one unit is influenced by other units. To tackle this challenge, we propose a machine learning-based method to identify and characterize heterogeneous network interference. Our approach accounts for latent complex network structures and automates the task of "exposure mapping'' determination, which addresses the two major limitations in the existing literature. We introduce "causal network motifs'' and employ transparent machine learning models to establish the most suitable exposure mapping that reflects underlying network interference patterns. Our method's efficacy has been validated through simulations on two synthetic experiments and a real-world, large-scale test involving 1-2 million Instagram users, outperforming conventional methods such as design-based cluster randomization and analysis-based neighborhood exposure mapping. 
    
[^6]: 关于超图上三体相互作用的非线性平均动力学收敛性的研究

    On the convergence of nonlinear averaging dynamics with three-body interactions on hypergraphs. (arXiv:2304.07203v1 [math.DS])

    [http://arxiv.org/abs/2304.07203](http://arxiv.org/abs/2304.07203)

    本文研究了超图上具有三体相互作用的离散时间非线性平均动力学，在初态和超图拓扑以及更新非线性相互作用下，产生了高阶动力效应。

    

    物理学、生物学和社会科学等领域的复杂网络系统通常涉及超出简单的成对相互作用的交互。超图作为描述和分析具有多体相互作用的系统复杂行为的强大建模工具。本文研究了具有三体相互作用的离散时间非线性平均动力学：底层超图由三元组作为超边界定相互作用的结构，而顶点通过加权的状态依赖的邻域对的状态平均更新其状态。相较于带有二体相互作用的图上的线性平均动力学，这个动力学不会收敛到初始状态的平均值，而是减少了初态和超图拓扑的复杂相互作用和更新的非线性产生高阶动力效应。

    Complex networked systems in fields such as physics, biology, and social sciences often involve interactions that extend beyond simple pairwise ones. Hypergraphs serve as powerful modeling tools for describing and analyzing the intricate behaviors of systems with multi-body interactions. Herein, we investigate a discrete-time nonlinear averaging dynamics with three-body interactions: an underlying hypergraph, comprising triples as hyperedges, delineates the structure of these interactions, while the vertices update their states through a weighted, state-dependent average of neighboring pairs' states. This dynamics captures reinforcing group effects, such as peer pressure, and exhibits higher-order dynamical effects resulting from a complex interplay between initial states, hypergraph topology, and nonlinearity of the update. Differently from linear averaging dynamics on graphs with two-body interactions, this model does not converge to the average of the initial states but rather induc
    

