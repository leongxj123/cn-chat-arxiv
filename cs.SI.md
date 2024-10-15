# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decoding Susceptibility: Modeling Misbelief to Misinformation Through a Computational Approach](https://arxiv.org/abs/2311.09630) | 通过计算方法对用户的潜在易感性水平进行建模，可以帮助理解易受错误信息影响的程度，为后续研究和应用提供重要参考。 |
| [^2] | [Pure Message Passing Can Estimate Common Neighbor for Link Prediction.](http://arxiv.org/abs/2309.00976) | 这篇论文提出了一种纯粹的消息传递方法，用于估计共同邻居进行链路预测。该方法通过利用输入向量的正交性来捕捉联合结构特征，提出了一种新的链路预测模型MPLP，该模型利用准正交向量估计链路级结构特征，同时保留了节点级复杂性。 |

# 详细

[^1]: 解码易感性：通过计算方法对错误信息进行建模

    Decoding Susceptibility: Modeling Misbelief to Misinformation Through a Computational Approach

    [https://arxiv.org/abs/2311.09630](https://arxiv.org/abs/2311.09630)

    通过计算方法对用户的潜在易感性水平进行建模，可以帮助理解易受错误信息影响的程度，为后续研究和应用提供重要参考。

    

    易受错误信息影响的程度描述了对不可验证主张的信仰程度，这是个体思维过程中的潜在因素，不可观察。现有易感性研究严重依赖于自我报告的信念，这可能存在偏见，收集成本高，并且难以在后续应用中扩展。为了解决这些限制，我们在这项研究中提出了一种计算方法来建模用户的潜在易感性水平。正如先前的研究所示，易感性受到各种因素的影响（例如人口统计因素、政治意识形态），并直接影响人们在社交媒体上的转发行为。为了表示基础心理过程，我们的易感性建模将这些因素作为输入，受到人们分享行为监督的引导。使用COVID-19作为实验领域，我们的实验证明了易感性评分之间存在显著的一致性。

    arXiv:2311.09630v2 Announce Type: replace  Abstract: Susceptibility to misinformation describes the degree of belief in unverifiable claims, a latent aspect of individuals' mental processes that is not observable. Existing susceptibility studies heavily rely on self-reported beliefs, which can be subject to bias, expensive to collect, and challenging to scale for downstream applications. To address these limitations, in this work, we propose a computational approach to model users' latent susceptibility levels. As shown in previous research, susceptibility is influenced by various factors (e.g., demographic factors, political ideology), and directly influences people's reposting behavior on social media. To represent the underlying mental process, our susceptibility modeling incorporates these factors as inputs, guided by the supervision of people's sharing behavior. Using COVID-19 as a testbed domain, our experiments demonstrate a significant alignment between the susceptibility score
    
[^2]: 纯粹的消息传递可以估计共同邻居进行链路预测

    Pure Message Passing Can Estimate Common Neighbor for Link Prediction. (arXiv:2309.00976v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.00976](http://arxiv.org/abs/2309.00976)

    这篇论文提出了一种纯粹的消息传递方法，用于估计共同邻居进行链路预测。该方法通过利用输入向量的正交性来捕捉联合结构特征，提出了一种新的链路预测模型MPLP，该模型利用准正交向量估计链路级结构特征，同时保留了节点级复杂性。

    

    消息传递神经网络（MPNN）已成为图表示学习中的事实标准。然而，在链路预测方面，它们往往表现不佳，被简单的启发式算法如共同邻居（CN）所超越。这种差异源于一个根本限制：尽管MPNN在节点级表示方面表现出色，但在编码链路预测中至关重要的联合结构特征（如CN）方面则遇到困难。为了弥合这一差距，我们认为通过利用输入向量的正交性，纯粹的消息传递确实可以捕捉到联合结构特征。具体而言，我们研究了MPNN在近似CN启发式算法方面的能力。基于我们的发现，我们引入了一种新的链路预测模型——消息传递链路预测器（MPLP）。MPLP利用准正交向量估计链路级结构特征，同时保留节点级复杂性。此外，我们的方法表明利用消息传递捕捉结构特征能够改善链路预测性能。

    Message Passing Neural Networks (MPNNs) have emerged as the {\em de facto} standard in graph representation learning. However, when it comes to link prediction, they often struggle, surpassed by simple heuristics such as Common Neighbor (CN). This discrepancy stems from a fundamental limitation: while MPNNs excel in node-level representation, they stumble with encoding the joint structural features essential to link prediction, like CN. To bridge this gap, we posit that, by harnessing the orthogonality of input vectors, pure message-passing can indeed capture joint structural features. Specifically, we study the proficiency of MPNNs in approximating CN heuristics. Based on our findings, we introduce the Message Passing Link Predictor (MPLP), a novel link prediction model. MPLP taps into quasi-orthogonal vectors to estimate link-level structural features, all while preserving the node-level complexities. Moreover, our approach demonstrates that leveraging message-passing to capture stru
    

