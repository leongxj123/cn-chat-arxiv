# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation](https://rss.arxiv.org/abs/2402.01542) | 本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。 |
| [^2] | [Layerwise complexity-matched learning yields an improved model of cortical area V2](https://arxiv.org/abs/2312.11436) | 通过分层复杂度匹配学习，我们开发了一种自下而上的自监督训练方法，最大化了特征相似性同时在不同位置的补丁上解除特征相关性。 |
| [^3] | [Sequential Model for Predicting Patient Adherence in Subcutaneous Immunotherapy for Allergic Rhinitis.](http://arxiv.org/abs/2401.11447) | 本研究利用新颖的机器学习模型，准确预测患者的非依从风险和相关的系统症状评分，为长期过敏性鼻炎亚卡激素皮下免疫治疗的管理提供了一种新的方法。 |
| [^4] | [Improved motif-scaffolding with SE(3) flow matching.](http://arxiv.org/abs/2401.04082) | 本文提出了一种使用SE(3)流匹配的图案支架方法，通过图案摊销和图案引导两种方法，可以生成结构上多样性更高的支架，与之前的最先进方法相比，成功率相当甚至更高。 |

# 详细

[^1]: 使用标记数据增强的测地插值方法学习蛋白质折叠的集体变量

    Learning Collective Variables for Protein Folding with Labeled Data Augmentation through Geodesic Interpolation

    [https://rss.arxiv.org/abs/2402.01542](https://rss.arxiv.org/abs/2402.01542)

    本实验提出了一种使用标记数据增强和测地插值方法学习蛋白质折叠的集体变量的策略，有效提高了采样效率，并在过渡态数据有限且嘈杂时表现优于基于分类器的方法。

    

    在分子动力学（MD）模拟中，通常通过增强采样技术来研究蛋白质折叠等罕见事件，其中大部分依赖于沿着加速发生的集体变量（CV）的定义。获得富有表达力的CV至关重要，但往往受到关于特定事件的信息不足的阻碍，例如从未折叠到折叠构象的转变。我们提出了一种模拟无关的数据增强策略，利用受物理启发的度量来生成类似蛋白质折叠转变的测地插值，从而提高采样效率，而无需真实的过渡态样本。通过利用插值进度参数，我们引入了基于回归的学习方案来构建CV模型，当过渡态数据有限且嘈杂时，该方法表现优于基于分类器的方法。

    In molecular dynamics (MD) simulations, rare events, such as protein folding, are typically studied by means of enhanced sampling techniques, most of which rely on the definition of a collective variable (CV) along which the acceleration occurs. Obtaining an expressive CV is crucial, but often hindered by the lack of information about the particular event, e.g., the transition from unfolded to folded conformation. We propose a simulation-free data augmentation strategy using physics-inspired metrics to generate geodesic interpolations resembling protein folding transitions, thereby improving sampling efficiency without true transition state samples. Leveraging interpolation progress parameters, we introduce a regression-based learning scheme for CV models, which outperforms classifier-based methods when transition state data is limited and noisy
    
[^2]: 分层复杂度匹配学习产生了改进的大脑皮层V2区模型

    Layerwise complexity-matched learning yields an improved model of cortical area V2

    [https://arxiv.org/abs/2312.11436](https://arxiv.org/abs/2312.11436)

    通过分层复杂度匹配学习，我们开发了一种自下而上的自监督训练方法，最大化了特征相似性同时在不同位置的补丁上解除特征相关性。

    

    人类识别复杂视觉模式的能力是通过顺次区域在腹侧视觉皮层中执行的变换所形成的。最近的端到端训练的深度神经网络逼近了人类的能力，并且提供了迄今为止对层次结构的后期神经反应的最佳描述。然而，与传统的手工设计模型相比，或者与优化编码效率或预测的模型相比，这些网络对前期阶段提供了较差的描述。此外，用于端到端学习的梯度反向传播通常被认为在生物上是不切实际的。在这里，我们通过开发一种自下而上的自监督训练方法，独立地作用于连续层，从而克服了这两个限制。具体地，我们最大化了对局部变形自然图像补丁对之间的特征相似性，并在采样自其他位置的补丁时使特征去相关。

    arXiv:2312.11436v2 Announce Type: replace-cross  Abstract: Human ability to recognize complex visual patterns arises through transformations performed by successive areas in the ventral visual cortex. Deep neural networks trained end-to-end for object recognition approach human capabilities, and offer the best descriptions to date of neural responses in the late stages of the hierarchy. But these networks provide a poor account of the early stages, compared to traditional hand-engineered models, or models optimized for coding efficiency or prediction. Moreover, the gradient backpropagation used in end-to-end learning is generally considered to be biologically implausible. Here, we overcome both of these limitations by developing a bottom-up self-supervised training methodology that operates independently on successive layers. Specifically, we maximize feature similarity between pairs of locally-deformed natural image patches, while decorrelating features across patches sampled from oth
    
[^3]: 预测过敏性鼻炎亚卡激素皮下免疫治疗中患者依从性的序列模型

    Sequential Model for Predicting Patient Adherence in Subcutaneous Immunotherapy for Allergic Rhinitis. (arXiv:2401.11447v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.11447](http://arxiv.org/abs/2401.11447)

    本研究利用新颖的机器学习模型，准确预测患者的非依从风险和相关的系统症状评分，为长期过敏性鼻炎亚卡激素皮下免疫治疗的管理提供了一种新的方法。

    

    目标：皮下免疫治疗(SCIT)是过敏性鼻炎的长效因果治疗。如何提高患者对变应原免疫治疗(AIT)的依从性以最大化治疗效果，在AIT管理中起着至关重要的作用。本研究旨在利用新颖的机器学习模型，准确预测患者的非依从风险和相关的系统症状评分，为长期AIT的管理提供一种新的方法。方法：本研究开发和分析了两种模型，序列潜在行为者-评论家模型(SLAC)和长短期记忆模型(LSTM)，并基于评分和依从性预测能力进行评估。结果：在排除第一时间步的偏倚样本后，SLAC模型的预测依从准确率为60%-72%，而LSTM模型的准确率为66%-84%，根据时间步长的不同而变化。SLAC模型的均方根误差(RMSE)范围在0.93到2.22之间，而LSTM模型的RMSE范围在...

    Objective: Subcutaneous Immunotherapy (SCIT) is the long-lasting causal treatment of allergic rhinitis. How to enhance the adherence of patients to maximize the benefit of allergen immunotherapy (AIT) plays a crucial role in the management of AIT. This study aims to leverage novel machine learning models to precisely predict the risk of non-adherence of patients and related systematic symptom scores, to provide a novel approach in the management of long-term AIT.  Methods: The research develops and analyzes two models, Sequential Latent Actor-Critic (SLAC) and Long Short-Term Memory (LSTM), evaluating them based on scoring and adherence prediction capabilities.  Results: Excluding the biased samples at the first time step, the predictive adherence accuracy of the SLAC models is from $60\,\%$ to $72\%$, and for LSTM models, it is $66\,\%$ to $84\,\%$, varying according to the time steps. The range of Root Mean Square Error (RMSE) for SLAC models is between $0.93$ and $2.22$, while for L
    
[^4]: 使用SE(3)流匹配改进了图案支架技术

    Improved motif-scaffolding with SE(3) flow matching. (arXiv:2401.04082v1 [q-bio.QM])

    [http://arxiv.org/abs/2401.04082](http://arxiv.org/abs/2401.04082)

    本文提出了一种使用SE(3)流匹配的图案支架方法，通过图案摊销和图案引导两种方法，可以生成结构上多样性更高的支架，与之前的最先进方法相比，成功率相当甚至更高。

    

    蛋白质设计通常从一个图案的期望功能开始，图案支架旨在构建一个功能性蛋白质。最近，生成模型在设计各种图案的支架方面取得了突破性的成功。然而，生成的支架往往缺乏结构多样性，这可能会影响湿实验验证的成功。在这项工作中，我们将FrameFlow，一种用于蛋白质主链生成的SE(3)流匹配模型扩展到使用两种互补的方法进行图案支架。第一种方法是图案摊销，即使用数据增强策略，将FrameFlow训练为以图案为输入。第二种方法是图案引导，它使用FrameFlow的条件分数估计进行支架构建，并且不需要额外的训练。这两种方法的成功率与之前的最先进方法相当或更高，并且可以产生结构上多样性更高2.5倍的支架。

    Protein design often begins with knowledge of a desired function from a motif which motif-scaffolding aims to construct a functional protein around. Recently, generative models have achieved breakthrough success in designing scaffolds for a diverse range of motifs. However, the generated scaffolds tend to lack structural diversity, which can hinder success in wet-lab validation. In this work, we extend FrameFlow, an SE(3) flow matching model for protein backbone generation, to perform motif-scaffolding with two complementary approaches. The first is motif amortization, in which FrameFlow is trained with the motif as input using a data augmentation strategy. The second is motif guidance, which performs scaffolding using an estimate of the conditional score from FrameFlow, and requires no additional training. Both approaches achieve an equivalent or higher success rate than previous state-of-the-art methods, with 2.5 times more structurally diverse scaffolds. Code: https://github.com/ mi
    

