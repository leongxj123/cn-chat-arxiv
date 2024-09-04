# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An embedding-based distance for temporal graphs.](http://arxiv.org/abs/2401.12843) | 本研究提出了一种基于图嵌入的时间图距离计算方法，能够有效区分具有不同结构和时间属性的图，适用于大规模时间图。 |
| [^2] | [Disentangling the Potential Impacts of Papers into Diffusion, Conformity, and Contribution Values.](http://arxiv.org/abs/2311.09262) | 这项研究提出了一种新颖的图神经网络（称为DPPDCC），用于将论文的潜在影响分解为传播、一致性和贡献值。通过编码时态和结构特征，捕捉知识流动，并使用对比增强图揭示流行度，进一步预测引用分组来建模一致性。应用正交约束来鼓励独特建模，并保留原始信息。 |

# 详细

[^1]: 基于嵌入距离计算的时间图

    An embedding-based distance for temporal graphs. (arXiv:2401.12843v1 [cs.SI])

    [http://arxiv.org/abs/2401.12843](http://arxiv.org/abs/2401.12843)

    本研究提出了一种基于图嵌入的时间图距离计算方法，能够有效区分具有不同结构和时间属性的图，适用于大规模时间图。

    

    我们基于使用时间尊重的随机游走构建的图嵌入来定义了一种时间图之间的距离。我们研究了匹配图和不匹配图的情况，当存在已知的节点关系时，以及当不存在该关系并且图可能具有不同的大小时的情况。通过使用真实和合成的时间网络数据，我们展示了我们所提出的距离定义的优势，表明它能够区分具有不同结构和时间属性的图。利用最先进的机器学习技术，我们提出了一种适用于大规模时间图的距离计算的高效实现。

    We define a distance between temporal graphs based on graph embeddings built using time-respecting random walks. We study both the case of matched graphs, when there exists a known relation between the nodes, and the unmatched case, when such a relation is unavailable and the graphs may be of different sizes. We illustrate the interest of our distance definition, using both real and synthetic temporal network data, by showing its ability to discriminate between graphs with different structural and temporal properties. Leveraging state-of-the-art machine learning techniques, we propose an efficient implementation of distance computation that is viable for large-scale temporal graphs.
    
[^2]: 将论文的潜在影响分解为传播、一致性和贡献值的研究

    Disentangling the Potential Impacts of Papers into Diffusion, Conformity, and Contribution Values. (arXiv:2311.09262v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2311.09262](http://arxiv.org/abs/2311.09262)

    这项研究提出了一种新颖的图神经网络（称为DPPDCC），用于将论文的潜在影响分解为传播、一致性和贡献值。通过编码时态和结构特征，捕捉知识流动，并使用对比增强图揭示流行度，进一步预测引用分组来建模一致性。应用正交约束来鼓励独特建模，并保留原始信息。

    

    论文的潜在影响受到多种因素的影响，包括其流行度和贡献。现有模型通常基于静态图来估计原始引用计数，未能从细微的角度区分价值。在本研究中，我们提出了一种新颖的图神经网络，用于将论文的潜在影响分解为传播、一致性和贡献值（称为DPPDCC）。给定一个目标论文，DPPDCC在构建的动态异构图中编码了时态和结构特征。特别地，为了捕捉知识流动，我们强调了论文之间的比较和共引/被引信息的重要性，并进行了快照演化的聚合。为了揭示流行度，我们通过对比增强图来提取传播的本质，并预测累积的引用分组以建模一致性。我们进一步应用正交约束来鼓励每个角度的独特建模，并保留其固有获得的信息。

    The potential impact of an academic paper is determined by various factors, including its popularity and contribution. Existing models usually estimate original citation counts based on static graphs and fail to differentiate values from nuanced perspectives. In this study, we propose a novel graph neural network to Disentangle the Potential impacts of Papers into Diffusion, Conformity, and Contribution values (called DPPDCC). Given a target paper, DPPDCC encodes temporal and structural features within the constructed dynamic heterogeneous graph. Particularly, to capture the knowledge flow, we emphasize the importance of comparative and co-cited/citing information between papers and aggregate snapshots evolutionarily. To unravel popularity, we contrast augmented graphs to extract the essence of diffusion and predict the accumulated citation binning to model conformity. We further apply orthogonal constraints to encourage distinct modeling of each perspective and preserve the inherent v
    

