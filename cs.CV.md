# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination.](http://arxiv.org/abs/2311.02960) | 本文通过研究中间特征的结构，揭示了深度网络在层级特征学习过程中的演化模式。研究发现线性层在特征学习中起到了与深层非线性网络类似的作用。 |
| [^3] | [GreatSplicing: A Semantically Rich Splicing Dataset.](http://arxiv.org/abs/2310.10070) | 本文提出了一个语义丰富的拼接数据集GreatSplicing，通过包括大量不同语义类别的拼接区域，训练的模型在拼接痕迹检测上表现出较低的误识率和更好的跨数据集检测能力。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 通过层间特征压缩和差别性学习理解深度表示学习

    Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination. (arXiv:2311.02960v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.02960](http://arxiv.org/abs/2311.02960)

    本文通过研究中间特征的结构，揭示了深度网络在层级特征学习过程中的演化模式。研究发现线性层在特征学习中起到了与深层非线性网络类似的作用。

    

    在过去的十年中，深度学习已经证明是从原始数据中学习有意义特征的一种高效工具。然而，深度网络如何在不同层级上进行等级特征学习仍然是一个开放问题。在这项工作中，我们试图通过研究中间特征的结构揭示这个谜团。受到我们实证发现的线性层在特征学习中模仿非线性网络中深层的角色的启发，我们研究了深度线性网络如何将输入数据转化为输出，通过研究训练后的每个层的输出（即特征）在多类分类问题的背景下。为了实现这个目标，我们首先定义了衡量中间特征的类内压缩和类间差别性的度量标准。通过对这两个度量标准的理论分析，我们展示了特征从浅层到深层的演变遵循着一种简单而量化的模式，前提是输入数据是

    Over the past decade, deep learning has proven to be a highly effective tool for learning meaningful features from raw data. However, it remains an open question how deep networks perform hierarchical feature learning across layers. In this work, we attempt to unveil this mystery by investigating the structures of intermediate features. Motivated by our empirical findings that linear layers mimic the roles of deep layers in nonlinear networks for feature learning, we explore how deep linear networks transform input data into output by investigating the output (i.e., features) of each layer after training in the context of multi-class classification problems. Toward this goal, we first define metrics to measure within-class compression and between-class discrimination of intermediate features, respectively. Through theoretical analysis of these two metrics, we show that the evolution of features follows a simple and quantitative pattern from shallow to deep layers when the input data is
    
[^3]: GreatSplicing: 一个语义丰富的拼接数据集

    GreatSplicing: A Semantically Rich Splicing Dataset. (arXiv:2310.10070v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.10070](http://arxiv.org/abs/2310.10070)

    本文提出了一个语义丰富的拼接数据集GreatSplicing，通过包括大量不同语义类别的拼接区域，训练的模型在拼接痕迹检测上表现出较低的误识率和更好的跨数据集检测能力。

    

    在现有的拼接伪造数据集中，拼接区域的语义变化不足导致训练的检测模型对语义特征的过拟合。同时，由于缺乏合理的数据集，不同的检测方法在实验设置上无法达成一致。为了解决这些紧迫的问题，本文提出了GreatSplicing，一个手动创建的具有大量和高质量的拼接数据集。GreatSplicing包括5000张拼接图像，并涵盖了335个不同的语义类别的拼接区域，让神经网络更好地抓住拼接痕迹。大量实验证明，使用GreatSplicing训练的模型相较于现有数据集表现出较低的误识率和更好的跨数据集检测能力。此外，GreatSplicing可供所有研究目的使用，并可从www.greatsplicing.net下载。

    In existing splicing forgery datasets, the insufficient semantic varieties of spliced regions cause a problem that trained detection models overfit semantic features rather than splicing traces. Meanwhile, because of the absence of a reasonable dataset, different detection methods proposed cannot reach a consensus on experimental settings. To address these urgent issues, GreatSplicing, a manually created splicing dataset with a considerable amount and high quality, is proposed in this paper. GreatSplicing comprises 5,000 spliced images and covers spliced regions with 335 distinct semantic categories, allowing neural networks to grasp splicing traces better. Extensive experiments demonstrate that models trained on GreatSplicing exhibit minimal misidentification rates and superior cross-dataset detection capabilities compared to existing datasets. Furthermore, GreatSplicing is available for all research purposes and can be downloaded from www.greatsplicing.net.
    

