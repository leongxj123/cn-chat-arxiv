# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test](https://arxiv.org/abs/2402.11089) | 通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。 |
| [^2] | [Unified Concept Editing in Diffusion Models.](http://arxiv.org/abs/2308.14761) | 本文提出了一种称为统一概念编辑（UCE）的方法，通过使用一个闭合解决方案对模型进行编辑，同时解决文本到图像模型中的偏见、版权和冒犯性内容等问题。实验证明了该方法的改进和可扩展性。 |
| [^3] | [Convolutions Through the Lens of Tensor Networks.](http://arxiv.org/abs/2307.02275) | 该论文提供了一种通过张量网络理解和演化卷积的新视角，可以通过绘制和操作张量网络来进行函数转换、子张量访问和融合。研究人员还演示了卷积图表的导出以及各种自动微分操作和二阶信息逼近图表的生成，同时还提供了特定于卷积的图表转换，以优化计算性能。 |
| [^4] | [Boosting Text-to-Image Diffusion Models with Fine-Grained Semantic Rewards.](http://arxiv.org/abs/2305.19599) | 本文提出了FineRewards，通过引入细粒度的语义奖励，即标题奖励和SAM奖励，来改进文本到图像扩散模型中文本和图像之间的对齐。 |
| [^5] | [ARBEx: Attentive Feature Extraction with Reliability Balancing for Robust Facial Expression Learning.](http://arxiv.org/abs/2305.01486) | 本论文提出了一个名为ARBEx的框架，它采用了可靠性平衡方法来应对面部表情学习任务中的数据偏差和不确定性。该框架还引入了可学习的锚点和多头自注意机制，并在多个公共数据集上取得了有效性验证。 |

# 详细

[^1]: 男性CEO和女性助理：通过成对刻板印象测试探究文本-图像模型中的性别偏见

    The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test

    [https://arxiv.org/abs/2402.11089](https://arxiv.org/abs/2402.11089)

    通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。

    

    最近大规模的文本到图像（T2I）模型（如DALLE-3）展示了在新应用中的巨大潜力，但也面临前所未有的公平挑战。先前的研究揭示了单人图像生成中的性别偏见，但T2I模型应用可能需要同时描绘两个或更多人。该设定中的潜在偏见仍未被探究，导致使用中的公平相关风险。为了研究T2I模型中性别偏见的基本方面，我们提出了一种新颖的成对刻板印象测试（PST）偏见评估框架。PST促使模型生成同一图像中的两个个体，用与相反性别刻板印象相关联的两个社会身份来描述他们。通过生成的图像遵从性别刻板印象的程度来衡量偏见。利用PST，我们从两个角度评估DALLE-3：性别职业中的偏见和组织权力中的偏见。

    arXiv:2402.11089v1 Announce Type: cross  Abstract: Recent large-scale Text-To-Image (T2I) models such as DALLE-3 demonstrate great potential in new applications, but also face unprecedented fairness challenges. Prior studies revealed gender biases in single-person image generation, but T2I model applications might require portraying two or more people simultaneously. Potential biases in this setting remain unexplored, leading to fairness-related risks in usage. To study these underlying facets of gender biases in T2I models, we propose a novel Paired Stereotype Test (PST) bias evaluation framework. PST prompts the model to generate two individuals in the same image. They are described with two social identities that are stereotypically associated with the opposite gender. Biases can then be measured by the level of conformation to gender stereotypes in generated images. Using PST, we evaluate DALLE-3 from 2 perspectives: biases in gendered occupation and biases in organizational power.
    
[^2]: 扩散模型中的统一概念编辑

    Unified Concept Editing in Diffusion Models. (arXiv:2308.14761v1 [cs.CV])

    [http://arxiv.org/abs/2308.14761](http://arxiv.org/abs/2308.14761)

    本文提出了一种称为统一概念编辑（UCE）的方法，通过使用一个闭合解决方案对模型进行编辑，同时解决文本到图像模型中的偏见、版权和冒犯性内容等问题。实验证明了该方法的改进和可扩展性。

    

    文本到图像模型存在各种安全问题，可能限制其适用性。先前的方法分别解决了文本到图像模型中的偏见、版权和冒犯性内容等各个问题。然而，在真实世界中，所有这些问题都同时出现在同一个模型中。我们提出了一种使用单一方法解决所有问题的方法，名为统一概念编辑（UCE）。我们的方法在不经过训练的情况下通过闭合解决方案对模型进行编辑，并可无缝地扩展到文本条件下的扩散模型上进行并行编辑。我们通过编辑文本到图像的投影来展示可扩展的同时去偏见、消除风格和内容调节，我们进行了大量实验证明了相对于先前方法的改进效果和可扩展性。我们的代码可以在https://unified.baulab.info上找到。

    Text-to-image models suffer from various safety issues that may limit their suitability for deployment. Previous methods have separately addressed individual issues of bias, copyright, and offensive content in text-to-image models. However, in the real world, all of these issues appear simultaneously in the same model. We present a method that tackles all issues with a single approach. Our method, Unified Concept Editing (UCE), edits the model without training using a closed-form solution, and scales seamlessly to concurrent edits on text-conditional diffusion models. We demonstrate scalable simultaneous debiasing, style erasure, and content moderation by editing text-to-image projections, and we present extensive experiments demonstrating improved efficacy and scalability over prior work. Our code is available at https://unified.baulab.info
    
[^3]: 透过张量网络的视角解析卷积

    Convolutions Through the Lens of Tensor Networks. (arXiv:2307.02275v1 [cs.LG])

    [http://arxiv.org/abs/2307.02275](http://arxiv.org/abs/2307.02275)

    该论文提供了一种通过张量网络理解和演化卷积的新视角，可以通过绘制和操作张量网络来进行函数转换、子张量访问和融合。研究人员还演示了卷积图表的导出以及各种自动微分操作和二阶信息逼近图表的生成，同时还提供了特定于卷积的图表转换，以优化计算性能。

    

    尽管卷积的直观概念简单，但其分析比稠密层更加复杂，这使得理论和算法的推广变得困难。我们通过张量网络（TN）提供了对卷积的新视角，通过绘制图表、操作图表进行函数转换、子张量访问和融合来推理底层张量乘法。我们通过推导各种自动微分操作的图表以及具有完整超参数支持、批处理、通道组和任意卷积维度泛化的流行的二阶信息逼近的图表来展示这种表达能力。此外，我们基于连接模式提供了特定于卷积的转换，允许在评估之前重新连接和简化图表。最后，我们通过依赖于高效TN缩并的已建立机制来探究计算性能。我们的TN实现加速了最近提出的

    Despite their simple intuition, convolutions are more tedious to analyze than dense layers, which complicates the generalization of theoretical and algorithmic ideas. We provide a new perspective onto convolutions through tensor networks (TNs) which allow reasoning about the underlying tensor multiplications by drawing diagrams, and manipulating them to perform function transformations, sub-tensor access, and fusion. We demonstrate this expressive power by deriving the diagrams of various autodiff operations and popular approximations of second-order information with full hyper-parameter support, batching, channel groups, and generalization to arbitrary convolution dimensions. Further, we provide convolution-specific transformations based on the connectivity pattern which allow to re-wire and simplify diagrams before evaluation. Finally, we probe computational performance, relying on established machinery for efficient TN contraction. Our TN implementation speeds up a recently-proposed
    
[^4]: 细粒度语义奖励增强文本到图像扩散模型

    Boosting Text-to-Image Diffusion Models with Fine-Grained Semantic Rewards. (arXiv:2305.19599v1 [cs.CV])

    [http://arxiv.org/abs/2305.19599](http://arxiv.org/abs/2305.19599)

    本文提出了FineRewards，通过引入细粒度的语义奖励，即标题奖励和SAM奖励，来改进文本到图像扩散模型中文本和图像之间的对齐。

    

    最近，文本到图像扩散模型的研究取得了显著的成功，在给定的文本提示下生成了高质量、逼真的图像。然而，由于缺乏细粒度语义指导，以成功诊断形态差异为止，以前的方法无法执行文本概念和生成的图像之间的准确形态对齐。在本文中，我们提出了FineRewards，通过引入两种新的细粒度语义奖励--标题奖励和语义分割任何事物（SAM）奖励，来改进文本到图像扩散模型中文本和图像之间的对齐。

    Recent advances in text-to-image diffusion models have achieved remarkable success in generating high-quality, realistic images from given text prompts. However, previous methods fail to perform accurate modality alignment between text concepts and generated images due to the lack of fine-level semantic guidance that successfully diagnoses the modality discrepancy. In this paper, we propose FineRewards to improve the alignment between text and images in text-to-image diffusion models by introducing two new fine-grained semantic rewards: the caption reward and the Semantic Segment Anything (SAM) reward. From the global semantic view, the caption reward generates a corresponding detailed caption that depicts all important contents in the synthetic image via a BLIP-2 model and then calculates the reward score by measuring the similarity between the generated caption and the given prompt. From the local semantic view, the SAM reward segments the generated images into local parts with categ
    
[^5]: ARBEx：用于鲁棒性面部表情学习的关注特征提取与可靠性平衡框架

    ARBEx: Attentive Feature Extraction with Reliability Balancing for Robust Facial Expression Learning. (arXiv:2305.01486v1 [cs.CV])

    [http://arxiv.org/abs/2305.01486](http://arxiv.org/abs/2305.01486)

    本论文提出了一个名为ARBEx的框架，它采用了可靠性平衡方法来应对面部表情学习任务中的数据偏差和不确定性。该框架还引入了可学习的锚点和多头自注意机制，并在多个公共数据集上取得了有效性验证。

    

    本论文提出了一个名为ARBEx的框架，它是由Vision Transformer驱动的新型关注特征提取框架，带有可靠性平衡，以应对面部表情学习任务中的较差类分布、偏差和不确定性。我们采用了多种数据预处理和精化方法以及基于窗口的交叉关注ViT来充分利用数据。我们还在嵌入空间中引入了可学习的锚点，加上标签分布和多头自注意机制，以通过可靠性平衡优化对弱预测的性能，这是一种提高标签预测韧性的策略。为了确保正确的标签分类并提高模型的区分能力，我们引入了锚损失，鼓励锚点之间的大间隔。另外，多头自注意机制也是可训练的，对于提升在FEL任务中的表现至关重要。最后，我们在多个公共数据集上验证了ARBEx的有效性。

    In this paper, we introduce a framework ARBEx, a novel attentive feature extraction framework driven by Vision Transformer with reliability balancing to cope against poor class distributions, bias, and uncertainty in the facial expression learning (FEL) task. We reinforce several data pre-processing and refinement methods along with a window-based cross-attention ViT to squeeze the best of the data. We also employ learnable anchor points in the embedding space with label distributions and multi-head self-attention mechanism to optimize performance against weak predictions with reliability balancing, which is a strategy that leverages anchor points, attention scores, and confidence values to enhance the resilience of label predictions. To ensure correct label classification and improve the models' discriminative power, we introduce anchor loss, which encourages large margins between anchor points. Additionally, the multi-head self-attention mechanism, which is also trainable, plays an i
    

