# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models](https://arxiv.org/abs/2402.06223) | 通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。 |
| [^2] | [Topologically-Regularized Multiple Instance Learning for Red Blood Cell Disease Classification.](http://arxiv.org/abs/2307.14025) | 本论文提出一种基于拓扑正则化的多实例学习方法，用于罕见贫血疾病的红细胞分类。通过从单个红细胞图像中提取多尺度的拓扑特征来进行模型正则化，以保持数据的特征拓扑属性。实验结果表明，该方法是有效的。 |

# 详细

[^1]: 通过潜在部分因果模型揭示多模式对比表示学习

    Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models

    [https://arxiv.org/abs/2402.06223](https://arxiv.org/abs/2402.06223)

    通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。

    

    多模式对比表示学习方法在各个领域取得了成功，部分原因是由于它们能够生成复杂现象的有意义的共享表示。为了增强对这些获得的表示的深度分析和理解，我们引入了一种特别针对多模态数据设计的统一因果模型。通过研究这个模型，我们展示了多模式对比表示学习在识别在提出的统一模型中的潜在耦合变量方面的优秀能力，即使在不同假设下导致的线性或置换变换。我们的发现揭示了预训练的多模态模型（如CLIP）通过线性独立分量分析这一令人惊讶的简单而高效的工具学习分离表示的潜力。实验证明了我们发现的鲁棒性，即使在被违反假设的情况下，也验证了所提出方法在学习疾病方面的有效性。

    Multimodal contrastive representation learning methods have proven successful across a range of domains, partly due to their ability to generate meaningful shared representations of complex phenomena. To enhance the depth of analysis and understanding of these acquired representations, we introduce a unified causal model specifically designed for multimodal data. By examining this model, we show that multimodal contrastive representation learning excels at identifying latent coupled variables within the proposed unified model, up to linear or permutation transformations resulting from different assumptions. Our findings illuminate the potential of pre-trained multimodal models, eg, CLIP, in learning disentangled representations through a surprisingly simple yet highly effective tool: linear independent component analysis. Experiments demonstrate the robustness of our findings, even when the assumptions are violated, and validate the effectiveness of the proposed method in learning dise
    
[^2]: 基于拓扑正则化的多实例学习用于红细胞疾病分类

    Topologically-Regularized Multiple Instance Learning for Red Blood Cell Disease Classification. (arXiv:2307.14025v1 [cs.LG])

    [http://arxiv.org/abs/2307.14025](http://arxiv.org/abs/2307.14025)

    本论文提出一种基于拓扑正则化的多实例学习方法，用于罕见贫血疾病的红细胞分类。通过从单个红细胞图像中提取多尺度的拓扑特征来进行模型正则化，以保持数据的特征拓扑属性。实验结果表明，该方法是有效的。

    

    使用显微图像诊断罕见的贫血疾病对于熟练的专家和机器学习方法来说都具有挑战性。由于在单个血样中有数千个与疾病相关的细胞，这构成了一个复杂的多实例学习（MIL）问题。虽然红细胞的空间邻域本身并不重要，但整个血样的拓扑结构，即数据的几何性质，包含了有益的特征，以解决典型的MIL问题，如梯度消失和在有限数据上训练时的过拟合。因此，我们开发了一种基于拓扑的方法，从单个红细胞图像的包中提取多尺度的拓扑特征。这些拓扑特征被用来对模型进行正则化，强制保持数据的特征拓扑属性。在包含71个罕见贫血疾病患者的数据集上，包括521张红细胞显微图像，我们的实验表明拓扑正则化是一个有效的方法。

    Diagnosing rare anemia disorders using microscopic images is challenging for skilled specialists and machine-learning methods alike. Due to thousands of disease-relevant cells in a single blood sample, this constitutes a complex multiple-instance learning (MIL) problem. While the spatial neighborhood of red blood cells is not meaningful per se, the topology, i.e., the geometry of blood samples as a whole, contains informative features to remedy typical MIL issues, such as vanishing gradients and overfitting when training on limited data. We thus develop a topology-based approach that extracts multi-scale topological features from bags of single red blood cell images. The topological features are used to regularize the model, enforcing the preservation of characteristic topological properties of the data. Applied to a dataset of 71 patients suffering from rare anemia disorders with 521 microscopic images of red blood cells, our experiments show that topological regularization is an effe
    

