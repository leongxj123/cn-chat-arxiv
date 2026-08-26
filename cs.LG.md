# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AdAdaGrad: Adaptive Batch Size Schemes for Adaptive Gradient Methods](https://arxiv.org/abs/2402.11215) | AdAdaGrad和AdAdaGradNorm是一个自适应增加批大小的方法，在深度学习中引入了自适应批大小策略，证明AdaGradNorm以高概率在$O(1/K)$速度下收敛。 |
| [^2] | [HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA](https://arxiv.org/abs/2402.01767) | HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。 |
| [^3] | [A Discriminative Latent-Variable Model for Bilingual Lexicon Induction](https://arxiv.org/abs/1808.09334) | 引入判别式潜变量模型，结合先前研究的词典先验和表示法，提出了用于双语词典归纳的新方法，并通过实验证据展示先验可以改善诱导的双语词典。 |
| [^4] | [RACR-MIL: Weakly Supervised Skin Cancer Grading using Rank-Aware Contextual Reasoning on Whole Slide Images.](http://arxiv.org/abs/2308.15618) | RACR-MIL是一个自动化的弱监督的皮肤癌分级方法，可以使用整张切片图像进行训练，无需细粒度的肿瘤注释，通过在切片图像中的瓦片章节中使用注意力机制的多实例学习，可以为切片图像分配分级。该方法主要创新包括使用空间和语义接近性定义切片图像图像以编码肿瘤区域的局部和非局部依赖关系以及使用序数排名约束保证注意力网络的性能 |

# 详细

[^1]: AdAdaGrad：自适应梯度方法的自适应批大小方案

    AdAdaGrad: Adaptive Batch Size Schemes for Adaptive Gradient Methods

    [https://arxiv.org/abs/2402.11215](https://arxiv.org/abs/2402.11215)

    AdAdaGrad和AdAdaGradNorm是一个自适应增加批大小的方法，在深度学习中引入了自适应批大小策略，证明AdaGradNorm以高概率在$O(1/K)$速度下收敛。

    

    随机梯度优化器中批量大小的选择对模型训练至关重要。然而，在训练过程中变化批大小的实践相对其他超参数较少探讨。我们研究了从自适应采样方法中导出的自适应批大小策略，传统上仅应用于随机梯度下降。考虑到学习速率和批大小之间的显著相互作用，以及自适应梯度方法在深度学习中的普及，我们强调在这些情境中需要自适应批大小策略。我们介绍了AdAdaGrad及其标量变体AdAdaGradNorm，它们在训练过程中逐渐增加批大小，同时使用AdaGrad和AdaGradNorm进行模型更新。我们证明了AdaGradNorm以高概率以$O(1/K)$的速度收敛，用于找到光滑非凸函数的一阶稳定点在$K$次迭代内。

    arXiv:2402.11215v1 Announce Type: new  Abstract: The choice of batch sizes in stochastic gradient optimizers is critical for model training. However, the practice of varying batch sizes throughout the training process is less explored compared to other hyperparameters. We investigate adaptive batch size strategies derived from adaptive sampling methods, traditionally applied only in stochastic gradient descent. Given the significant interplay between learning rates and batch sizes, and considering the prevalence of adaptive gradient methods in deep learning, we emphasize the need for adaptive batch size strategies in these contexts. We introduce AdAdaGrad and its scalar variant AdAdaGradNorm, which incrementally increase batch sizes during training, while model updates are performed using AdaGrad and AdaGradNorm. We prove that AdaGradNorm converges with high probability at a rate of $\mathscr{O}(1/K)$ for finding a first-order stationary point of smooth nonconvex functions within $K$ i
    
[^2]: HiQA：一种用于大规模文档问答的分层上下文增强的RAG模型

    HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA

    [https://arxiv.org/abs/2402.01767](https://arxiv.org/abs/2402.01767)

    HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。

    

    随着利用外部工具的语言模型代理迅速发展，使用补充文档和检索增强生成（RAG）方法的问答（QA）方法学取得了重要进展。这种进步提高了语言模型的回答质量，并减轻了幻觉的出现。然而，当面临大量无法区分的文档时，这些方法在检索准确性方面表现有限，给实际应用带来了显著挑战。针对这些新兴的挑战，我们提出了HiQA，这是一个先进的多文档问答（MDQA）框架，将级联的元数据整合到内容中，同时具备多路径检索机制。我们还发布了一个名为MasQA的基准来评估和研究MDQA。最后，HiQA在多文档环境中展示了最先进的性能。

    As language model agents leveraging external tools rapidly evolve, significant progress has been made in question-answering(QA) methodologies utilizing supplementary documents and the Retrieval-Augmented Generation (RAG) approach. This advancement has improved the response quality of language models and alleviates the appearance of hallucination. However, these methods exhibit limited retrieval accuracy when faced with massive indistinguishable documents, presenting notable challenges in their practical application. In response to these emerging challenges, we present HiQA, an advanced framework for multi-document question-answering (MDQA) that integrates cascading metadata into content as well as a multi-route retrieval mechanism. We also release a benchmark called MasQA to evaluate and research in MDQA. Finally, HiQA demonstrates the state-of-the-art performance in multi-document environments.
    
[^3]: 一种用于双语词典归纳的判别式潜变量模型

    A Discriminative Latent-Variable Model for Bilingual Lexicon Induction

    [https://arxiv.org/abs/1808.09334](https://arxiv.org/abs/1808.09334)

    引入判别式潜变量模型，结合先前研究的词典先验和表示法，提出了用于双语词典归纳的新方法，并通过实验证据展示先验可以改善诱导的双语词典。

    

    我们引入了一种新颖的用于双语词典归纳的判别式潜变量模型。我们的模型将Haghighi等人（2008）的二分匹配词典先验与基于表示的方法（Artetxe等人，2017）相结合。为了训练模型，我们推导出了高效的Viterbi EM算法。我们在两个度量标准下对六种语言对进行了实证结果，并显示先验改善了诱导的双语词典。我们还演示了如何将先前的工作视为类似风格的潜变量模型，尽管有不同的先验。

    arXiv:1808.09334v3 Announce Type: replace  Abstract: We introduce a novel discriminative latent variable model for bilingual lexicon induction. Our model combines the bipartite matching dictionary prior of Haghighi et al. (2008) with a representation-based approach (Artetxe et al., 2017). To train the model, we derive an efficient Viterbi EM algorithm. We provide empirical results on six language pairs under two metrics and show that the prior improves the induced bilingual lexicons. We also demonstrate how previous work may be viewed as a similarly fashioned latent-variable model, albeit with a different prior.
    
[^4]: RACR-MIL：使用基于排名感知背景推理的整张切片图像进行弱监督的皮肤癌分级

    RACR-MIL: Weakly Supervised Skin Cancer Grading using Rank-Aware Contextual Reasoning on Whole Slide Images. (arXiv:2308.15618v1 [cs.CV])

    [http://arxiv.org/abs/2308.15618](http://arxiv.org/abs/2308.15618)

    RACR-MIL是一个自动化的弱监督的皮肤癌分级方法，可以使用整张切片图像进行训练，无需细粒度的肿瘤注释，通过在切片图像中的瓦片章节中使用注意力机制的多实例学习，可以为切片图像分配分级。该方法主要创新包括使用空间和语义接近性定义切片图像图像以编码肿瘤区域的局部和非局部依赖关系以及使用序数排名约束保证注意力网络的性能

    

    Cutaneous squamous cell cancer (cSCC) is the second most common skin cancer in the US. It is diagnosed by manual multi-class tumor grading using a tissue whole slide image (WSI), which is subjective and suffers from inter-pathologist variability. We propose an automated weakly-supervised grading approach for cSCC WSIs that is trained using WSI-level grade and does not require fine-grained tumor annotations. The proposed model, RACR-MIL, transforms each WSI into a bag of tiled patches and leverages attention-based multiple-instance learning to assign a WSI-level grade. We propose three key innovations to address general as well as cSCC-specific challenges in tumor grading. First, we leverage spatial and semantic proximity to define a WSI graph that encodes both local and non-local dependencies between tumor regions and leverage graph attention convolution to derive contextual patch features. Second, we introduce a novel ordinal ranking constraint on the patch attention network to ensure

    Cutaneous squamous cell cancer (cSCC) is the second most common skin cancer in the US. It is diagnosed by manual multi-class tumor grading using a tissue whole slide image (WSI), which is subjective and suffers from inter-pathologist variability. We propose an automated weakly-supervised grading approach for cSCC WSIs that is trained using WSI-level grade and does not require fine-grained tumor annotations. The proposed model, RACR-MIL, transforms each WSI into a bag of tiled patches and leverages attention-based multiple-instance learning to assign a WSI-level grade. We propose three key innovations to address general as well as cSCC-specific challenges in tumor grading. First, we leverage spatial and semantic proximity to define a WSI graph that encodes both local and non-local dependencies between tumor regions and leverage graph attention convolution to derive contextual patch features. Second, we introduce a novel ordinal ranking constraint on the patch attention network to ensure
    

