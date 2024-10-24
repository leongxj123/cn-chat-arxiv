# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Few-Shot Adversarial Prompt Learning on Vision-Language Models](https://arxiv.org/abs/2403.14774) | 本文提出了一个少样本对抗提示框架，在视觉-语言模型中通过有限数据调整输入序列，显著提升对抗鲁棒性，并通过端到端学习对抗性相关的文本监督。 |
| [^2] | [Fine-tuning with Very Large Dropout](https://arxiv.org/abs/2403.00946) | 通过使用非常高的dropout率进行微调，可以实现超出分布性能，这超出了集成和权重平均方法。 |
| [^3] | [Real-World Robot Applications of Foundation Models: A Review](https://arxiv.org/abs/2402.05741) | 本文综述了基础模型在真实世界机器人中的应用，重点是替换现有机器人系统中的特定组件。这些基础模型在输入输出关系、感知、运动规划和控制等方面扮演了重要角色。未来的挑战和对实际机器人应用的影响也被讨论到。 |
| [^4] | [Your Diffusion Model is Secretly a Certifiably Robust Classifier](https://arxiv.org/abs/2402.02316) | 这项研究提出了一种新的扩散分类器家族，称为噪声扩散分类器（NDCs），其具有最新的可证明的鲁棒性。通过将扩散分类器推广到分类高斯受损数据，并将其与随机平滑技术相结合，构建了具有非常量Lipschitzness的平滑分类器。这些NDCs显示出卓越的认证鲁棒性。 |
| [^5] | [Retrieving Conditions from Reference Images for Diffusion Models](https://arxiv.org/abs/2312.02521) | 本文将主题驱动生成视为扩散模型中的一个统一检索问题，引入了一种名为RetriNet的新颖扩散模型架构，通过精确检索主题属性并过滤无关信息来解决问题，在人脸生成方面表现出卓越性能。 |
| [^6] | [Conquering the Communication Constraints to Enable Large Pre-Trained Models in Federated Learning](https://arxiv.org/abs/2210.01708) | 研究克服联邦学习中通信约束的方法，以实现强大的预训练模型在FL中的应用，并同时减少通信负担。 |
| [^7] | [Boosting Text-to-Image Diffusion Models with Fine-Grained Semantic Rewards.](http://arxiv.org/abs/2305.19599) | 本文提出了FineRewards，通过引入细粒度的语义奖励，即标题奖励和SAM奖励，来改进文本到图像扩散模型中文本和图像之间的对齐。 |
| [^8] | [ARBEx: Attentive Feature Extraction with Reliability Balancing for Robust Facial Expression Learning.](http://arxiv.org/abs/2305.01486) | 本论文提出了一个名为ARBEx的框架，它采用了可靠性平衡方法来应对面部表情学习任务中的数据偏差和不确定性。该框架还引入了可学习的锚点和多头自注意机制，并在多个公共数据集上取得了有效性验证。 |

# 详细

[^1]: 视觉-语言模型上的少样本对抗提示学习

    Few-Shot Adversarial Prompt Learning on Vision-Language Models

    [https://arxiv.org/abs/2403.14774](https://arxiv.org/abs/2403.14774)

    本文提出了一个少样本对抗提示框架，在视觉-语言模型中通过有限数据调整输入序列，显著提升对抗鲁棒性，并通过端到端学习对抗性相关的文本监督。

    

    深度神经网络对于微不可见的对抗性扰动的脆弱性已经引起了广泛关注。受到视觉-语言基础模型成功的启发，先前的努力通过将对抗性视觉特征与文本监督对齐来实现零样本对抗鲁棒性。但在实践中，由于包括重大适应成本、次优文本监督和未受控制的自然泛化能力在内的多个问题，它们仍然不尽人意。为了解决这些问题，本文提出了一个少样本对抗提示框架，通过有限的数据调整输入序列使得对抗鲁棒性得到显著提升。具体而言，我们通过提供对抗相关的文本监督，该监督是从对抗性示例中端到端学习的，来实现这一点。我们还提出了一个增强多模态特征一致性并鼓励不同

    arXiv:2403.14774v1 Announce Type: cross  Abstract: The vulnerability of deep neural networks to imperceptible adversarial perturbations has attracted widespread attention. Inspired by the success of vision-language foundation models, previous efforts achieved zero-shot adversarial robustness by aligning adversarial visual features with text supervision. However, in practice, they are still unsatisfactory due to several issues, including heavy adaptation cost, suboptimal text supervision, and uncontrolled natural generalization capacity. In this paper, to address these issues, we propose a few-shot adversarial prompt framework where adapting input sequences with limited data makes significant adversarial robustness improvement. Specifically, we achieve this by providing adversarially correlated text supervision that is end-to-end learned from adversarial examples. We also propose a novel training objective that enhances the consistency of multi-modal features while encourages differenti
    
[^2]: 使用非常大的Dropout进行微调

    Fine-tuning with Very Large Dropout

    [https://arxiv.org/abs/2403.00946](https://arxiv.org/abs/2403.00946)

    通过使用非常高的dropout率进行微调，可以实现超出分布性能，这超出了集成和权重平均方法。

    

    今天不可能假装机器学习实践与训练和测试数据遵循相同分布的观念是兼容的。该论文调查了使用非常高的丢弃率来获得这种丰富表示，尽管使用这样的丢弃率从头开始训练深度网络几乎是不可能的，但在这些条件下对大型预训练模型进行微调不仅是可能的，而且实现了超越集成和权重平均方法的超出分布性能。

    arXiv:2403.00946v1 Announce Type: new  Abstract: It is impossible today to pretend that the practice of machine learning is compatible with the idea that training and testing data follow the same distribution. Several authors have recently used ensemble techniques to show how scenarios involving multiple data distributions are best served by representations that are both richer than those obtained by regularizing for the best in-distribution performance, and richer than those obtained under the influence of the implicit sparsity bias of common stochastic gradient procedures.   This contribution investigates the use of very high dropout rates instead of ensembles to obtain such rich representations. Although training a deep network from scratch using such dropout rates is virtually impossible, fine-tuning a large pre-trained model under such conditions is not only possible but also achieves out-of-distribution performances that exceed those of both ensembles and weight averaging methods
    
[^3]: 基于基础模型的真实世界机器人应用：一项综述

    Real-World Robot Applications of Foundation Models: A Review

    [https://arxiv.org/abs/2402.05741](https://arxiv.org/abs/2402.05741)

    本文综述了基础模型在真实世界机器人中的应用，重点是替换现有机器人系统中的特定组件。这些基础模型在输入输出关系、感知、运动规划和控制等方面扮演了重要角色。未来的挑战和对实际机器人应用的影响也被讨论到。

    

    最近，基于大规模语言模型（LLMs）和视觉语言模型（VLMs）等基础模型的发展，通过对大量数据的训练，为不同任务和模态的灵活应用提供了便利。它们的影响涵盖了包括医疗、教育和机器人等各个领域。本文概述了基础模型在真实世界机器人中的实际应用情况，重点是替换现有机器人系统中的特定组件。总结涵盖了基础模型中的输入输出关系以及它们在机器人领域中的感知、运动规划和控制等方面的作用。本文还讨论了未来挑战和对实际机器人应用的影响。

    Recent developments in foundation models, like Large Language Models (LLMs) and Vision-Language Models (VLMs), trained on extensive data, facilitate flexible application across different tasks and modalities. Their impact spans various fields, including healthcare, education, and robotics. This paper provides an overview of the practical application of foundation models in real-world robotics, with a primary emphasis on the replacement of specific components within existing robot systems. The summary encompasses the perspective of input-output relationships in foundation models, as well as their role in perception, motion planning, and control within the field of robotics. This paper concludes with a discussion of future challenges and implications for practical robot applications.
    
[^4]: 你的扩散模型实际上是一个可证明鲁棒的分类器

    Your Diffusion Model is Secretly a Certifiably Robust Classifier

    [https://arxiv.org/abs/2402.02316](https://arxiv.org/abs/2402.02316)

    这项研究提出了一种新的扩散分类器家族，称为噪声扩散分类器（NDCs），其具有最新的可证明的鲁棒性。通过将扩散分类器推广到分类高斯受损数据，并将其与随机平滑技术相结合，构建了具有非常量Lipschitzness的平滑分类器。这些NDCs显示出卓越的认证鲁棒性。

    

    近期，扩散模型被作为鲁棒分类的生成器分类器所采用。然而，对于扩散分类器鲁棒性的综合理论理解仍然缺乏，这让我们怀疑它们是否会容易受到未来更强攻击的影响。在本研究中，我们提出了一种新的扩散分类器家族，命名为噪声扩散分类器（NDCs），其具有最新的可证明的鲁棒性。具体来说，我们通过推导这些分布的证据下界（ELBOs），利用ELBO近似似然度量，并使用贝叶斯定理计算分类概率，将扩散分类器推广到分类高斯受损数据。我们将这些推广的扩散分类器与随机平滑技术相结合，构建具有非常量Lipschitzness的平滑分类器。实验结果表明我们提出的NDCs在鲁棒性方面具有卓越的认证能力。值得注意的是，我们是第一个达到80%的...

    Diffusion models are recently employed as generative classifiers for robust classification. However, a comprehensive theoretical understanding of the robustness of diffusion classifiers is still lacking, leading us to question whether they will be vulnerable to future stronger attacks. In this study, we propose a new family of diffusion classifiers, named Noised Diffusion Classifiers~(NDCs), that possess state-of-the-art certified robustness. Specifically, we generalize the diffusion classifiers to classify Gaussian-corrupted data by deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. We integrate these generalized diffusion classifiers with randomized smoothing to construct smoothed classifiers possessing non-constant Lipschitzness. Experimental results demonstrate the superior certified robustness of our proposed NDCs. Notably, we are the first to achieve 80\%
    
[^5]: 从参考图像中检索条件用于扩散模型

    Retrieving Conditions from Reference Images for Diffusion Models

    [https://arxiv.org/abs/2312.02521](https://arxiv.org/abs/2312.02521)

    本文将主题驱动生成视为扩散模型中的一个统一检索问题，引入了一种名为RetriNet的新颖扩散模型架构，通过精确检索主题属性并过滤无关信息来解决问题，在人脸生成方面表现出卓越性能。

    

    新开发的基于扩散的技术展示了在生成各种高质量图像方面的卓越能力，引起了各种应用的相当大兴趣。一个普遍的场景是基于参考图像中的一个主题生成新的图像。这个主题可以是风格化头像的面部身份，虚拟试穿的身体和服装等。满足这一要求正在演变成一门称为主题驱动生成的领域。在本文中，我们将主题驱动生成视为扩散模型中的一个统一检索问题。我们引入了一种名为RetriNet的新颖扩散模型架构，旨在通过精确地从参考图像中检索主题属性并过滤掉无关信息来解决这些问题。与现有的最先进方法相比，RetriNet在人脸生成方面表现出令人印象深刻的性能。我们进一步提出了一个研究和...

    arXiv:2312.02521v2 Announce Type: replace-cross  Abstract: Newly developed diffusion-based techniques have showcased phenomenal abilities in producing a wide range of high-quality images, sparking considerable interest in various applications. A prevalent scenario is to generate new images based on a subject from reference images. This subject could be face identity for styled avatars, body and clothing for virtual try-on and so on. Satisfying this requirement is evolving into a field called Subject-Driven Generation. In this paper, we consider Subject-Driven Generation as a unified retrieval problem with diffusion models. We introduce a novel diffusion model architecture, named RetriNet, designed to address and solve these problems by retrieving subject attributes from reference images precisely, and filter out irrelevant information. RetriNet demonstrates impressive performance when compared to existing state-of-the-art approaches in face generation. We further propose a research and
    
[^6]: 克服通信约束，实现联邦学习中大型预训练模型的应用

    Conquering the Communication Constraints to Enable Large Pre-Trained Models in Federated Learning

    [https://arxiv.org/abs/2210.01708](https://arxiv.org/abs/2210.01708)

    研究克服联邦学习中通信约束的方法，以实现强大的预训练模型在FL中的应用，并同时减少通信负担。

    

    联邦学习（FL）已经成为一种旨在在本地设备上协力训练模型而不需要对原始数据进行中心化访问的有前景的范式。在典型的FL范式（例如FedAvg）中，每一轮模型权重都会被发送到参与客户端并回传到服务器。最近，在联邦学习优化和收敛改进方面展示了使用小型预训练模型是有效的。然而，最近的最先进预训练模型变得更加强大，但也拥有更多参数。在传统的FL中，共享巨大的模型权重可以迅速给系统带来巨大的通信负担，尤其是如果采用更加强大的模型。我们能否找到一个解决方案，在FL中启用这些强大且现成的预训练模型以实现出色性能的同时减少通信负担？为此，我们研究了使用参数高效的方法

    arXiv:2210.01708v3 Announce Type: replace  Abstract: Federated learning (FL) has emerged as a promising paradigm for enabling the collaborative training of models without centralized access to the raw data on local devices. In the typical FL paradigm (e.g., FedAvg), model weights are sent to and from the server each round to participating clients. Recently, the use of small pre-trained models has been shown effective in federated learning optimization and improving convergence. However, recent state-of-the-art pre-trained models are getting more capable but also have more parameters. In conventional FL, sharing the enormous model weights can quickly put a massive communication burden on the system, especially if more capable models are employed. Can we find a solution to enable those strong and readily-available pre-trained models in FL to achieve excellent performance while simultaneously reducing the communication burden? To this end, we investigate the use of parameter-efficient fin
    
[^7]: 细粒度语义奖励增强文本到图像扩散模型

    Boosting Text-to-Image Diffusion Models with Fine-Grained Semantic Rewards. (arXiv:2305.19599v1 [cs.CV])

    [http://arxiv.org/abs/2305.19599](http://arxiv.org/abs/2305.19599)

    本文提出了FineRewards，通过引入细粒度的语义奖励，即标题奖励和SAM奖励，来改进文本到图像扩散模型中文本和图像之间的对齐。

    

    最近，文本到图像扩散模型的研究取得了显著的成功，在给定的文本提示下生成了高质量、逼真的图像。然而，由于缺乏细粒度语义指导，以成功诊断形态差异为止，以前的方法无法执行文本概念和生成的图像之间的准确形态对齐。在本文中，我们提出了FineRewards，通过引入两种新的细粒度语义奖励--标题奖励和语义分割任何事物（SAM）奖励，来改进文本到图像扩散模型中文本和图像之间的对齐。

    Recent advances in text-to-image diffusion models have achieved remarkable success in generating high-quality, realistic images from given text prompts. However, previous methods fail to perform accurate modality alignment between text concepts and generated images due to the lack of fine-level semantic guidance that successfully diagnoses the modality discrepancy. In this paper, we propose FineRewards to improve the alignment between text and images in text-to-image diffusion models by introducing two new fine-grained semantic rewards: the caption reward and the Semantic Segment Anything (SAM) reward. From the global semantic view, the caption reward generates a corresponding detailed caption that depicts all important contents in the synthetic image via a BLIP-2 model and then calculates the reward score by measuring the similarity between the generated caption and the given prompt. From the local semantic view, the SAM reward segments the generated images into local parts with categ
    
[^8]: ARBEx：用于鲁棒性面部表情学习的关注特征提取与可靠性平衡框架

    ARBEx: Attentive Feature Extraction with Reliability Balancing for Robust Facial Expression Learning. (arXiv:2305.01486v1 [cs.CV])

    [http://arxiv.org/abs/2305.01486](http://arxiv.org/abs/2305.01486)

    本论文提出了一个名为ARBEx的框架，它采用了可靠性平衡方法来应对面部表情学习任务中的数据偏差和不确定性。该框架还引入了可学习的锚点和多头自注意机制，并在多个公共数据集上取得了有效性验证。

    

    本论文提出了一个名为ARBEx的框架，它是由Vision Transformer驱动的新型关注特征提取框架，带有可靠性平衡，以应对面部表情学习任务中的较差类分布、偏差和不确定性。我们采用了多种数据预处理和精化方法以及基于窗口的交叉关注ViT来充分利用数据。我们还在嵌入空间中引入了可学习的锚点，加上标签分布和多头自注意机制，以通过可靠性平衡优化对弱预测的性能，这是一种提高标签预测韧性的策略。为了确保正确的标签分类并提高模型的区分能力，我们引入了锚损失，鼓励锚点之间的大间隔。另外，多头自注意机制也是可训练的，对于提升在FEL任务中的表现至关重要。最后，我们在多个公共数据集上验证了ARBEx的有效性。

    In this paper, we introduce a framework ARBEx, a novel attentive feature extraction framework driven by Vision Transformer with reliability balancing to cope against poor class distributions, bias, and uncertainty in the facial expression learning (FEL) task. We reinforce several data pre-processing and refinement methods along with a window-based cross-attention ViT to squeeze the best of the data. We also employ learnable anchor points in the embedding space with label distributions and multi-head self-attention mechanism to optimize performance against weak predictions with reliability balancing, which is a strategy that leverages anchor points, attention scores, and confidence values to enhance the resilience of label predictions. To ensure correct label classification and improve the models' discriminative power, we introduce anchor loss, which encourages large margins between anchor points. Additionally, the multi-head self-attention mechanism, which is also trainable, plays an i
    

