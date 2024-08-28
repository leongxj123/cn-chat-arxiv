# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education](https://arxiv.org/abs/2402.06264) | 本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。 |
| [^2] | [Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs.](http://arxiv.org/abs/2401.13054) | 本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。 |

# 详细

[^1]: LLaVA-Docent：利用多模态大型语言模型支持艺术鉴赏教育的教学调优

    LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education

    [https://arxiv.org/abs/2402.06264](https://arxiv.org/abs/2402.06264)

    本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。

    

    艺术鉴赏对于培养学习者的批判性思维和情感智力至关重要。然而，传统的艺术鉴赏教育常面临艺术资源有限的问题，特别是对于弱势学生，并且在主流教育中过度强调科学技术工程和数学科目。为了应对这些挑战，最近的技术进步为创新解决方案铺平了道路。本研究探索了多模态大型语言模型（MLLM）在艺术鉴赏教育中的应用，重点是开发了LLaVA-Docent模型来利用这些进展。我们的方法包括全面的文献综述和与领域专家的咨询，从而形成了一个强大的数据框架。利用这个框架，我们生成了一个虚拟对话数据集，该数据集被GPT-4利用。这个数据集对于训练MLLM（即LLaVA-Docent）起到了关键作用。六名研究人员进行了定量和定性评估。

    Art appreciation is vital in nurturing critical thinking and emotional intelligence among learners. However, traditional art appreciation education has often been hindered by limited access to art resources, especially for disadvantaged students, and an imbalanced emphasis on STEM subjects in mainstream education. In response to these challenges, recent technological advancements have paved the way for innovative solutions. This study explores the application of multi-modal large language models (MLLMs) in art appreciation education, focusing on developing LLaVA-Docent, a model that leverages these advancements. Our approach involved a comprehensive literature review and consultations with experts in the field, leading to developing a robust data framework. Utilizing this framework, we generated a virtual dialogue dataset that was leveraged by GPT-4. This dataset was instrumental in training the MLLM, named LLaVA-Docent. Six researchers conducted quantitative and qualitative evaluation
    
[^2]: 无计算困难的快速计算超图节点距离的方法

    Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs. (arXiv:2401.13054v1 [cs.SI])

    [http://arxiv.org/abs/2401.13054](http://arxiv.org/abs/2401.13054)

    本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。

    

    超图是图的推广，当考虑实体间的属性共享时会自然产生。尽管可以通过将超边扩展为完全连接的子图来将超图转换为图，但逆向操作在计算上非常复杂且属于NP-complete问题。因此，我们假设超图包含比图更多的信息。此外，直接操作超图比将其扩展为图更为方便。超图中的一个开放问题是如何精确高效地计算节点之间的距离。通过估计节点距离，我们能够找到节点的最近邻居，并使用K最近邻（KNN）方法在超图上执行标签传播。在本文中，我们提出了一种基于随机游走的新方法，实现了在超图上进行标签传播。我们将节点距离估计为随机游走的预期到达时间。我们注意到简单随机游走（SRW）无法准确描述节点之间的距离，因此我们引入了"frustrated"的概念。

    A hypergraph is a generalization of a graph that arises naturally when attribute-sharing among entities is considered. Although a hypergraph can be converted into a graph by expanding its hyperedges into fully connected subgraphs, going the reverse way is computationally complex and NP-complete. We therefore hypothesize that a hypergraph contains more information than a graph. In addition, it is more convenient to manipulate a hypergraph directly, rather than expand it into a graph. An open problem in hypergraphs is how to accurately and efficiently calculate their node distances. Estimating node distances enables us to find a node's nearest neighbors, and perform label propagation on hypergraphs using a K-nearest neighbors (KNN) approach. In this paper, we propose a novel approach based on random walks to achieve label propagation on hypergraphs. We estimate node distances as the expected hitting times of random walks. We note that simple random walks (SRW) cannot accurately describe 
    

