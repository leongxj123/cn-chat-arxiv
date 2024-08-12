# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Domain Generalization through Meta-Learning: A Survey](https://arxiv.org/abs/2404.02785) | 元学习是一种有前景的方法，通过获取可转移知识实现在各种任务之间快速适应，为解决深度神经网络在面对分布变化和有限标记数据时泛化能力不佳提供了新途径。 |
| [^2] | [IllusionVQA: A Challenging Optical Illusion Dataset for Vision Language Models](https://arxiv.org/abs/2403.15952) | 提出了IllusionVQA数据集，用于测试视觉语言模型在错觉和难解场景下的表现，研究发现在理解任务和定位任务上，表现最佳的VLM为GPT4V，而人类表现更胜一筹。 |
| [^3] | [Rehabilitation Exercise Quality Assessment through Supervised Contrastive Learning with Hard and Soft Negatives](https://arxiv.org/abs/2403.02772) | 这项研究引入了一种新的有监督对比学习框架，通过结合硬和软负样本，有效地解决了康复锻炼评估中样本稀缺的问题 |
| [^4] | [Diffusion Reward: Learning Rewards via Conditional Video Diffusion](https://arxiv.org/abs/2312.14134) | 通过条件视频扩散学习奖励，解决复杂视觉强化学习问题，有效提高了任务成功率，超越了基线方法。 |
| [^5] | [General Lipschitz: Certified Robustness Against Resolvable Semantic Transformations via Transformation-Dependent Randomized Smoothing.](http://arxiv.org/abs/2309.16710) | 我们提出了一种新的框架“一般李普希茨（GL）”，用于对神经网络进行认证，以抵御可组合的可解释的语义干扰。方法在ImageNet数据集上与最先进的方法表现相当。 |
| [^6] | [MVMR: Evaluating Natural Language Video Localization Bias over Multiple Reliable Videos Pool.](http://arxiv.org/abs/2309.16701) | 本文提出了一个名为MVMR的任务，旨在给定文本查询从大量视频集中定位视频帧。我们通过已有数据集进行相似性筛选来构建数据集，并引入三个MVMR数据集。我们采用了嵌入式文本相似度匹配和视频-语言对齐技术来计算相关性得分，并为MVMR任务开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。 |
| [^7] | [Pair then Relation: Pair-Net for Panoptic Scene Graph Generation.](http://arxiv.org/abs/2307.08699) | 本文提出了一种名为Pair-Net的全景场景图生成框架，通过使用配对提案网络（PPN）来学习和过滤主体和物体之间的稀疏配对关系，解决了当前全景场景图生成方法中忽视的对象间配对回忆率问题。 |

# 详细

[^1]: 通过元学习实现领域泛化：一项调查

    Domain Generalization through Meta-Learning: A Survey

    [https://arxiv.org/abs/2404.02785](https://arxiv.org/abs/2404.02785)

    元学习是一种有前景的方法，通过获取可转移知识实现在各种任务之间快速适应，为解决深度神经网络在面对分布变化和有限标记数据时泛化能力不佳提供了新途径。

    

    深度神经网络(DNNs)已经彻底改变了人工智能，但是当面对分布之外(out-of-distribution, OOD)数据时往往表现不佳，这是因为在现实世界应用中由于领域转移不可避免，训练和测试数据被假定为共享相同分布的常见情况。尽管DNNs在大量数据和计算能力方面非常有效，但它们很难应对分布变化和有限标记数据，导致过拟合和跨不同任务和领域的泛化能力不佳。元学习提供了一种有前途的方法，通过采用能够在各种任务之间获取可转移知识的算法进行快速适应，从而消除了需要从头学习每个任务的必要性。本调查论文深入探讨了元学习领域，重点关注其对领域泛化的贡献。

    arXiv:2404.02785v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) have revolutionized artificial intelligence but often lack performance when faced with out-of-distribution (OOD) data, a common scenario due to the inevitable domain shifts in real-world applications. This limitation stems from the common assumption that training and testing data share the same distribution-an assumption frequently violated in practice. Despite their effectiveness with large amounts of data and computational power, DNNs struggle with distributional shifts and limited labeled data, leading to overfitting and poor generalization across various tasks and domains. Meta-learning presents a promising approach by employing algorithms that acquire transferable knowledge across various tasks for fast adaptation, eliminating the need to learn each task from scratch. This survey paper delves into the realm of meta-learning with a focus on its contribution to domain generalization. We first clarify the 
    
[^2]: IllusionVQA：一个挑战视觉语言模型的错觉数据集

    IllusionVQA: A Challenging Optical Illusion Dataset for Vision Language Models

    [https://arxiv.org/abs/2403.15952](https://arxiv.org/abs/2403.15952)

    提出了IllusionVQA数据集，用于测试视觉语言模型在错觉和难解场景下的表现，研究发现在理解任务和定位任务上，表现最佳的VLM为GPT4V，而人类表现更胜一筹。

    

    视觉语言模型（VLM）的出现使研究人员能够使用自然语言调查神经网络的视觉理解。 VLM不仅能够进行对象分类和检测，还能够进行视觉理解和常识推理。 这自然而然地引出了一个问题：当图像本身是不合理的时，VLM会如何回应？ 为此，我们提出了IllusionVQA：一个包含具有挑战性的光学错觉和难以解释的场景的多样数据集，以测试VLM在两种不同的多选VQA任务 - 理解和软定位的能力。 表现最佳的VLM GPT4V在理解任务（4-shot）上实现了62.99％的准确率，在定位任务（4-shot和Chain-of-Thought）上实现了49.7％的准确率。 人类评估表明，人类在理解和定位方面的准确率分别为91.03％和100％。 我们发现，在上下文学习（ICL）和Chain-of-Thought推理方面有很大帮助。

    arXiv:2403.15952v1 Announce Type: cross  Abstract: The advent of Vision Language Models (VLM) has allowed researchers to investigate the visual understanding of a neural network using natural language. Beyond object classification and detection, VLMs are capable of visual comprehension and common-sense reasoning. This naturally led to the question: How do VLMs respond when the image itself is inherently unreasonable? To this end, we present IllusionVQA: a diverse dataset of challenging optical illusions and hard-to-interpret scenes to test the capability of VLMs in two distinct multiple-choice VQA tasks - comprehension and soft localization. GPT4V, the best-performing VLM, achieves 62.99% accuracy (4-shot) on the comprehension task and 49.7% on the localization task (4-shot and Chain-of-Thought). Human evaluation reveals that humans achieve 91.03% and 100% accuracy in comprehension and localization. We discover that In-Context Learning (ICL) and Chain-of-Thought reasoning substantially
    
[^3]: 通过有监督对比学习进行康复锻炼质量评估，结合硬负样本和软负样本

    Rehabilitation Exercise Quality Assessment through Supervised Contrastive Learning with Hard and Soft Negatives

    [https://arxiv.org/abs/2403.02772](https://arxiv.org/abs/2403.02772)

    这项研究引入了一种新的有监督对比学习框架，通过结合硬和软负样本，有效地解决了康复锻炼评估中样本稀缺的问题

    

    基于锻炼的康复计划已被证明在提高生活质量、降低死亡率和再住院率方面是有效的。利用人工智能驱动的虚拟康复，患者可以在家独立完成锻炼，利用AI算法分析锻炼数据，为患者提供反馈，并向临床医生更新他们的进展情况。这些计划通常会指定各种锻炼类型，这导致康复锻炼评估数据集面临独特挑战：虽然在整体训练样本中丰富，但这些数据集通常对每种具体锻练类型的样本数量有限。这种差异影响了现有方法训练具有小样本量的每种锻练的可泛化模型的能力。为了解决这个问题，我们的论文引入了一种新颖的带有硬负样本和软负样本的有监督对比学习框架，有效利用了整个

    arXiv:2403.02772v1 Announce Type: cross  Abstract: Exercise-based rehabilitation programs have proven to be effective in enhancing the quality of life and reducing mortality and rehospitalization rates. AI-driven virtual rehabilitation, which allows patients to independently complete exercises at home, utilizes AI algorithms to analyze exercise data, providing feedback to patients and updating clinicians on their progress. These programs commonly prescribe a variety of exercise types, leading to a distinct challenge in rehabilitation exercise assessment datasets: while abundant in overall training samples, these datasets often have a limited number of samples for each individual exercise type. This disparity hampers the ability of existing approaches to train generalizable models with such a small sample size per exercise. Addressing this issue, our paper introduces a novel supervised contrastive learning framework with hard and soft negative samples that effectively utilizes the entir
    
[^4]: 通过条件视频扩散学习奖励

    Diffusion Reward: Learning Rewards via Conditional Video Diffusion

    [https://arxiv.org/abs/2312.14134](https://arxiv.org/abs/2312.14134)

    通过条件视频扩散学习奖励，解决复杂视觉强化学习问题，有效提高了任务成功率，超越了基线方法。

    

    从专家视频中学习奖励为强化学习任务指定预期行为提供了一种经济高效的解决方案。本文提出了Diffusion Reward，这是一个通过条件视频扩散模型从专家视频中学习奖励以解决复杂视觉强化学习问题的新框架。我们的关键见解是，在专家轨迹的条件下观察到较低的生成多样性。因此，Diffusion Reward被形式化为负的条件熵，鼓励专家式行为的有效探索。我们展示了我们的方法在MetaWorld和Adroit的10个机器人操纵任务中以视觉输入和稀疏奖励的有效性。此外，Diffusion Reward甚至能够成功有效地解决未见过的任务，大大超越了基线方法。项目页面和代码：https://diffusion-reward.github.io/。

    arXiv:2312.14134v2 Announce Type: replace  Abstract: Learning rewards from expert videos offers an affordable and effective solution to specify the intended behaviors for reinforcement learning tasks. In this work, we propose Diffusion Reward, a novel framework that learns rewards from expert videos via conditional video diffusion models for solving complex visual RL problems. Our key insight is that lower generative diversity is observed when conditioned on expert trajectories. Diffusion Reward is accordingly formalized by the negative of conditional entropy that encourages productive exploration of expert-like behaviors. We show the efficacy of our method over 10 robotic manipulation tasks from MetaWorld and Adroit with visual input and sparse reward. Moreover, Diffusion Reward could even solve unseen tasks successfully and effectively, largely surpassing baseline methods. Project page and code: https://diffusion-reward.github.io/.
    
[^5]: 通过依赖于变换的随机平滑，提供一般李普希茨：针对可解释的语义变换的认证健壮性

    General Lipschitz: Certified Robustness Against Resolvable Semantic Transformations via Transformation-Dependent Randomized Smoothing. (arXiv:2309.16710v1 [cs.CV])

    [http://arxiv.org/abs/2309.16710](http://arxiv.org/abs/2309.16710)

    我们提出了一种新的框架“一般李普希茨（GL）”，用于对神经网络进行认证，以抵御可组合的可解释的语义干扰。方法在ImageNet数据集上与最先进的方法表现相当。

    

    随机平滑是目前构建图像分类器的最先进方法，可以证明其对于有界干扰的抗性。然而，构建针对语义变换（例如，图像模糊、平移、Gamma矫正）及其组合的合理证书更为复杂。在本工作中，我们提出了一种新的框架“一般李普希茨（GL）”，用于对神经网络进行认证，以抵御可组合的可解释的语义干扰。在该框架中，我们分析了平滑分类器与变换参数之间的依赖关系，并推导出相应的健壮性证书。我们的方法在ImageNet数据集上与最先进的方法表现相当。

    Randomized smoothing is the state-of-the-art approach to construct image classifiers that are provably robust against additive adversarial perturbations of bounded magnitude. However, it is more complicated to construct reasonable certificates against semantic transformation (e.g., image blurring, translation, gamma correction) and their compositions. In this work, we propose \emph{General Lipschitz (GL),} a new framework to certify neural networks against composable resolvable semantic perturbations. Within the framework, we analyze transformation-dependent Lipschitz-continuity of smoothed classifiers w.r.t. transformation parameters and derive corresponding robustness certificates. Our method performs comparably to state-of-the-art approaches on the ImageNet dataset.
    
[^6]: MVMR: 在多个可靠视频集中评估自然语言视频定位偏差

    MVMR: Evaluating Natural Language Video Localization Bias over Multiple Reliable Videos Pool. (arXiv:2309.16701v1 [cs.CV])

    [http://arxiv.org/abs/2309.16701](http://arxiv.org/abs/2309.16701)

    本文提出了一个名为MVMR的任务，旨在给定文本查询从大量视频集中定位视频帧。我们通过已有数据集进行相似性筛选来构建数据集，并引入三个MVMR数据集。我们采用了嵌入式文本相似度匹配和视频-语言对齐技术来计算相关性得分，并为MVMR任务开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。

    

    随着近年来多媒体内容的激增，自然语言视频定位成为一个关键问题，它致力于检测与给定自然语言查询匹配的视频片段。然而，以往的研究都没有探索在存在多个正负视频的大量语料库中定位一个时刻。本文提出了一个名为MVMR（Massive Videos Moment Retrieval）的任务，旨在给定文本查询从大量视频集中定位视频帧。对于这个任务，我们提出了一种通过对现有视频定位数据集进行相似性筛选来构建数据集的方法，并引入了三个MVMR数据集。具体来说，我们采用基于嵌入的文本相似度匹配和视频-语言对齐技术来计算目标查询与视频之间的相关性得分，从而定义正负集。针对提出的MVMR任务，我们进一步开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。

    With the explosion of multimedia content in recent years, natural language video localization, which focuses on detecting video moment that matches a given natural language query, has become a critical problem. However, none of the previous research explores localizing a moment from a large corpus where multiple positive and negative videos exist. In this paper, we propose an MVMR (Massive Videos Moment Retrieval) task, which aims to localize video frames from a massive set of videos given a text query. For this task, we suggest methods for constructing datasets by employing similarity filtering on the existing video localization datasets and introduce three MVMR datasets. Specifically, we employ embedding-based text similarity matching and video-language grounding techniques to calculate the relevance score between a target query and videos to define positive and negative sets. For the proposed MVMR task, we further develop a strong model, Reliable Mutual Matching Network (RMMN), whic
    
[^7]: 聚焦场景图生成的配对-关系：基于配对网络的全景场景图生成

    Pair then Relation: Pair-Net for Panoptic Scene Graph Generation. (arXiv:2307.08699v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.08699](http://arxiv.org/abs/2307.08699)

    本文提出了一种名为Pair-Net的全景场景图生成框架，通过使用配对提案网络（PPN）来学习和过滤主体和物体之间的稀疏配对关系，解决了当前全景场景图生成方法中忽视的对象间配对回忆率问题。

    

    全景场景图生成是场景图生成中的一项挑战性任务，其旨在使用全景分割代替边界框来创建更全面的场景图表示。与场景图生成相比，全景场景图生成具有一些具有挑战性的问题：像素级分割输出和完全关系探索（它还考虑了物体和物质之间的关系）。因此，当前的全景场景图生成方法的性能有限，这阻碍了下游任务或应用的发展。本工作的目标是设计一种新颖且强大的全景场景图生成基准。为了实现这一目标，我们首先进行了深入分析，确定了当前全景场景图生成模型的瓶颈，发现对象间配对的回忆率是先前的全景场景图生成方法所忽视的一个关键因素。基于这一发现和最近的基于查询的框架，我们提出了一种新的框架：配对-关系（Pair-Net），它使用配对提案网络（PPN）来学习和过滤主体和物体之间的稀疏配对关系。

    Panoptic Scene Graph (PSG) is a challenging task in Scene Graph Generation (SGG) that aims to create a more comprehensive scene graph representation using panoptic segmentation instead of boxes. Compared to SGG, PSG has several challenging problems: pixel-level segment outputs and full relationship exploration (It also considers thing and stuff relation). Thus, current PSG methods have limited performance, which hinders downstream tasks or applications. The goal of this work aims to design a novel and strong baseline for PSG. To achieve that, we first conduct an in-depth analysis to identify the bottleneck of the current PSG models, finding that inter-object pair-wise recall is a crucial factor that was ignored by previous PSG methods. Based on this and the recent query-based frameworks, we present a novel framework: Pair then Relation (Pair-Net), which uses a Pair Proposal Network (PPN) to learn and filter sparse pair-wise relationships between subjects and objects. Moreover, we also 
    

