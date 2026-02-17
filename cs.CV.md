# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization](https://arxiv.org/abs/2403.15605) | 引入了一种新颖的FedDG架构方法gPerXAN，通过规范化方案和引导正则化器配合工作，实现了个性化显式组装规范化，有助于客户端模型对领域特征进行有选择性过滤。 |
| [^2] | [A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot.](http://arxiv.org/abs/2307.14397) | 本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，并提出了关于任务和方法的分类体系，研究了它们之间的相互作用，并探讨了未来的研究方向。 |
| [^3] | [TKN: Transformer-based Keypoint Prediction Network For Real-time Video Prediction.](http://arxiv.org/abs/2303.09807) | TKN是一种基于Transformer的实时视频预测解决方案，通过受限信息提取和并行预测方案来提升预测过程的速度，具有更高的精度和更低的计算成本。 |

# 详细

[^1]: 为联邦领域泛化高效组合规范化层与正则化方法

    Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization

    [https://arxiv.org/abs/2403.15605](https://arxiv.org/abs/2403.15605)

    引入了一种新颖的FedDG架构方法gPerXAN，通过规范化方案和引导正则化器配合工作，实现了个性化显式组装规范化，有助于客户端模型对领域特征进行有选择性过滤。

    

    领域转移是机器学习中一个严峻的问题，会导致模型在未知领域测试时性能下降。联邦领域泛化（FedDG）旨在以隐私保护的方式使用协作客户端训练全局模型，能够很好地泛化到可能存在领域转移的未知客户端。然而，大多数现有的FedDG方法可能会导致额外的数据泄露隐私风险，或者在客户端通信和计算成本方面产生显著开销，这在联邦学习范式中是主要关注的问题。为了解决这些挑战，我们引入了一种新颖的FedDG架构方法，即gPerXAN，它依赖于一个规范化方案与引导正则化器配合工作。具体来说，我们精心设计了个性化显式组装规范化，以强制客户端模型有选择地过滤对本地数据有偏向的特定领域特征。

    arXiv:2403.15605v1 Announce Type: cross  Abstract: Domain shift is a formidable issue in Machine Learning that causes a model to suffer from performance degradation when tested on unseen domains. Federated Domain Generalization (FedDG) attempts to train a global model using collaborative clients in a privacy-preserving manner that can generalize well to unseen clients possibly with domain shift. However, most existing FedDG methods either cause additional privacy risks of data leakage or induce significant costs in client communication and computation, which are major concerns in the Federated Learning paradigm. To circumvent these challenges, here we introduce a novel architectural method for FedDG, namely gPerXAN, which relies on a normalization scheme working with a guiding regularizer. In particular, we carefully design Personalized eXplicitly Assembled Normalization to enforce client models selectively filtering domain-specific features that are biased towards local data while ret
    
[^2]: 关于有限数据、少样本和零样本情况下生成建模的调查

    A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot. (arXiv:2307.14397v1 [cs.CV])

    [http://arxiv.org/abs/2307.14397](http://arxiv.org/abs/2307.14397)

    本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，并提出了关于任务和方法的分类体系，研究了它们之间的相互作用，并探讨了未来的研究方向。

    

    在机器学习中，生成建模旨在学习生成与训练数据分布统计相似的新数据。本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，称为数据约束下的生成建模（GM-DC）。这是一个重要的主题，当数据获取具有挑战性时，例如医疗应用。我们讨论了背景、挑战，并提出了两个分类体系：一个是GM-DC任务分类，另一个是GM-DC方法分类。重要的是，我们研究了不同GM-DC任务和方法之间的相互作用。此外，我们还强调了研究空白、研究趋势和未来探索的潜在途径。项目网站：https://gmdc-survey.github.io。

    In machine learning, generative modeling aims to learn to generate new data statistically similar to the training data distribution. In this paper, we survey learning generative models under limited data, few shots and zero shot, referred to as Generative Modeling under Data Constraint (GM-DC). This is an important topic when data acquisition is challenging, e.g. healthcare applications. We discuss background, challenges, and propose two taxonomies: one on GM-DC tasks and another on GM-DC approaches. Importantly, we study interactions between different GM-DC tasks and approaches. Furthermore, we highlight research gaps, research trends, and potential avenues for future exploration. Project website: https://gmdc-survey.github.io.
    
[^3]: TKN：基于Transformer的实时视频关键点预测网络

    TKN: Transformer-based Keypoint Prediction Network For Real-time Video Prediction. (arXiv:2303.09807v1 [cs.CV])

    [http://arxiv.org/abs/2303.09807](http://arxiv.org/abs/2303.09807)

    TKN是一种基于Transformer的实时视频预测解决方案，通过受限信息提取和并行预测方案来提升预测过程的速度，具有更高的精度和更低的计算成本。

    

    视频预测是一项具有广泛用途的复杂时间序列预测任务。然而，传统方法过于强调准确性，忽视了由于过于复杂的模型结构而导致的较慢的预测速度以及过多的冗余信息学习和GPU内存消耗。因此，我们提出了TKN，一种基于Transformer的关键点预测神经网络，通过受限信息提取和并行预测方案来提升预测过程的速度。TKN是我们目前所知的第一个实时视频预测解决方案，同时显著降低计算成本并保持其他性能。在KTH和Human Action 3D数据集上的大量实验表明，TKN在预测准确性和速度方面均优于现有的基准线。

    Video prediction is a complex time-series forecasting task with great potential in many use cases. However, conventional methods overemphasize accuracy while ignoring the slow prediction speed caused by complicated model structures that learn too much redundant information with excessive GPU memory consumption. Furthermore, conventional methods mostly predict frames sequentially (frame-by-frame) and thus are hard to accelerate. Consequently, valuable use cases such as real-time danger prediction and warning cannot achieve fast enough inference speed to be applicable in reality. Therefore, we propose a transformer-based keypoint prediction neural network (TKN), an unsupervised learning method that boost the prediction process via constrained information extraction and parallel prediction scheme. TKN is the first real-time video prediction solution to our best knowledge, while significantly reducing computation costs and maintaining other performance. Extensive experiments on KTH and Hum
    

