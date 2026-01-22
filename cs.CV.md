# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sora as an AGI World Model? A Complete Survey on Text-to-Video Generation](https://arxiv.org/abs/2403.05131) | 对文本到视频生成技术的发展进行了详细调查, 着重介绍了从传统生成模型到尖端Sora模型的转变，强调了可扩展性和通用性的发展。 |
| [^2] | [Benchmark data to study the influence of pre-training on explanation performance in MR image classification.](http://arxiv.org/abs/2306.12150) | 本研究提出了一个MRI分类任务的基准数据集，用于评估不同模型的解释性能。实验结果表明，XAI方法并不一定比简单模型提供更好的解释，且CNN的解释能力取决于底层数据的复杂性和标签的质量。 |

# 详细

[^1]: Sora作为AGI世界模型？关于文本到视频生成的完整调查

    Sora as an AGI World Model? A Complete Survey on Text-to-Video Generation

    [https://arxiv.org/abs/2403.05131](https://arxiv.org/abs/2403.05131)

    对文本到视频生成技术的发展进行了详细调查, 着重介绍了从传统生成模型到尖端Sora模型的转变，强调了可扩展性和通用性的发展。

    

    arXiv:2403.05131v1 公告类型: 新摘要: 文本到视频生成标志着生成式人工智能不断发展领域中的重要前沿，整合了文本到图像合成、视频字幕和文本引导编辑的进展。本调查对文本到视频技术的发展进行了批判性审视，重点关注传统生成模型向尖端Sora模型转变的过程，突出了可扩展性和通用性的发展。区别于以往作品的分析，我们深入探讨了这些模型的技术框架和演化路径。此外，我们还深入探讨了实际应用，并解决了伦理和技术挑战，如无法执行多实体处理、理解因果关系学习、理解物理互动、感知物体缩放和比例以及对抗物体幻觉，这也是生成模型中长期存在的问题。

    arXiv:2403.05131v1 Announce Type: new  Abstract: Text-to-video generation marks a significant frontier in the rapidly evolving domain of generative AI, integrating advancements in text-to-image synthesis, video captioning, and text-guided editing. This survey critically examines the progression of text-to-video technologies, focusing on the shift from traditional generative models to the cutting-edge Sora model, highlighting developments in scalability and generalizability. Distinguishing our analysis from prior works, we offer an in-depth exploration of the technological frameworks and evolutionary pathways of these models. Additionally, we delve into practical applications and address ethical and technological challenges such as the inability to perform multiple entity handling, comprehend causal-effect learning, understand physical interaction, perceive object scaling and proportioning, and combat object hallucination which is also a long-standing problem in generative models. Our c
    
[^2]: 基于预训练的影响因素研究医学图像分类解释性能的基准数据

    Benchmark data to study the influence of pre-training on explanation performance in MR image classification. (arXiv:2306.12150v1 [cs.CV])

    [http://arxiv.org/abs/2306.12150](http://arxiv.org/abs/2306.12150)

    本研究提出了一个MRI分类任务的基准数据集，用于评估不同模型的解释性能。实验结果表明，XAI方法并不一定比简单模型提供更好的解释，且CNN的解释能力取决于底层数据的复杂性和标签的质量。

    

    卷积神经网络（CNN）常常在医学预测任务中被成功地应用，通常与迁移学习相结合，在训练数据不足时能够提高性能。然而，由于CNN产生的模型高度复杂且通常不提供任何有关其预测机制的信息，这促使了“可解释性”人工智能（XAI）领域的研究。本文提出了一个基准数据集，用于在MRI分类任务中定量评估解释性能。通过这个基准数据集，我们可以了解迁移学习对解释质量的影响。实验结果表明，应用于基于迁移学习的CNN的流行XAI方法并不一定比简单模型提供更好的解释，并且CNN提供有意义解释的能力严重依赖于底层数据的复杂性和标签的质量。

    Convolutional Neural Networks (CNNs) are frequently and successfully used in medical prediction tasks. They are often used in combination with transfer learning, leading to improved performance when training data for the task are scarce. The resulting models are highly complex and typically do not provide any insight into their predictive mechanisms, motivating the field of 'explainable' artificial intelligence (XAI). However, previous studies have rarely quantitatively evaluated the 'explanation performance' of XAI methods against ground-truth data, and transfer learning and its influence on objective measures of explanation performance has not been investigated. Here, we propose a benchmark dataset that allows for quantifying explanation performance in a realistic magnetic resonance imaging (MRI) classification task. We employ this benchmark to understand the influence of transfer learning on the quality of explanations. Experimental results show that popular XAI methods applied to t
    

