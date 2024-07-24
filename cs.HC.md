# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Will the Real Linda Please Stand up...to Large Language Models? Examining the Representativeness Heuristic in LLMs](https://arxiv.org/abs/2404.01461) | 该研究调查了代表性启发式对大型语言模型推理的影响，并创建了专门的数据集进行实验验证 |
| [^2] | [DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video.](http://arxiv.org/abs/2303.13397) | 提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。 |

# 详细

[^1]: 请真正的琳达站出来...面对大语言模型？在LLMs中审视代表性启发式

    Will the Real Linda Please Stand up...to Large Language Models? Examining the Representativeness Heuristic in LLMs

    [https://arxiv.org/abs/2404.01461](https://arxiv.org/abs/2404.01461)

    该研究调查了代表性启发式对大型语言模型推理的影响，并创建了专门的数据集进行实验验证

    

    尽管大型语言模型（LLMs）在理解文本和生成类似人类文本方面表现出色，但它们可能会展现出从训练数据中获得的偏见。具体而言，LLMs可能会容易受到人类决策中的一种常见认知陷阱影响，即代表性启发式。这是心理学中的一个概念，指的是根据事件与一个众所周知的原型或典型例子的相似程度来判断事件发生的可能性，而不考虑更广泛的事实或统计证据。本研究调查了代表性启发式对LLM推理的影响。我们创建了REHEAT（Representativeness Heuristic AI Testing），一个包含涵盖六种常见代表性启发式类型问题的数据集。实验显示，应用于REHEAT的四个LLMs都表现出代表性启发式偏见。我们进一步确定了模型的推理步骤

    arXiv:2404.01461v1 Announce Type: new  Abstract: Although large language models (LLMs) have demonstrated remarkable proficiency in understanding text and generating human-like text, they may exhibit biases acquired from training data in doing so. Specifically, LLMs may be susceptible to a common cognitive trap in human decision-making called the representativeness heuristic. This is a concept in psychology that refers to judging the likelihood of an event based on how closely it resembles a well-known prototype or typical example versus considering broader facts or statistical evidence. This work investigates the impact of the representativeness heuristic on LLM reasoning. We created REHEAT (Representativeness Heuristic AI Testing), a dataset containing a series of problems spanning six common types of representativeness heuristics. Experiments reveal that four LLMs applied to REHEAT all exhibited representativeness heuristic biases. We further identify that the model's reasoning steps
    
[^2]: DDT：一种基于扩散驱动变压器的从视频中恢复人体网格的框架

    DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video. (arXiv:2303.13397v1 [cs.CV])

    [http://arxiv.org/abs/2303.13397](http://arxiv.org/abs/2303.13397)

    提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。

    

    人体网格恢复（HMR）为各种实际应用提供了丰富的人体信息，例如游戏、人机交互和虚拟现实。与单一图像方法相比，基于视频的方法可以利用时间信息通过融合人体运动先验进一步提高性能。然而，像 VIBE 这样的多对多方法存在运动平滑性和时间一致性的挑战。而像 TCMR 和 MPS-Net 这样的多对一方法则依赖于未来帧，在推理过程中是非因果和时间效率低下的。为了解决这些挑战，提出了一种新的基于扩散驱动变压器的视频 HMR 框架（DDT）。DDT 旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性。作为一种多对多方法，DDT 的解码器输出所有帧的人体网格，使 DDT 更适用于时间效率至关重要的实际应用。

    Human mesh recovery (HMR) provides rich human body information for various real-world applications such as gaming, human-computer interaction, and virtual reality. Compared to single image-based methods, video-based methods can utilize temporal information to further improve performance by incorporating human body motion priors. However, many-to-many approaches such as VIBE suffer from motion smoothness and temporal inconsistency. While many-to-one approaches such as TCMR and MPS-Net rely on the future frames, which is non-causal and time inefficient during inference. To address these challenges, a novel Diffusion-Driven Transformer-based framework (DDT) for video-based HMR is presented. DDT is designed to decode specific motion patterns from the input sequence, enhancing motion smoothness and temporal consistency. As a many-to-many approach, the decoder of our DDT outputs the human mesh of all the frames, making DDT more viable for real-world applications where time efficiency is cruc
    

