# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models](https://arxiv.org/abs/2403.12952) | 引入了测试时间原型转移（TPS）框架，通过动态学习每个原型的转移向量，有效地弥合了领域差距并增强了类 |
| [^2] | [Fusing Domain-Specific Content from Large Language Models into Knowledge Graphs for Enhanced Zero Shot Object State Classification](https://arxiv.org/abs/2403.12151) | 大型语言模型与知识图谱结合，提高零样本对象状态分类性能 |
| [^3] | [ChatGPT and biometrics: an assessment of face recognition, gender detection, and age estimation capabilities](https://arxiv.org/abs/2403.02965) | 本文评估了ChatGPT在面部识别、性别检测和年龄估计等生物识别任务中的表现，结果显示ChatGPT在面部识别方面具有较高准确性，并在性别检测方面表现显著，在年龄估计任务中也具有相当准确性。 |
| [^4] | [Weighted Ensemble Models Are Strong Continual Learners](https://arxiv.org/abs/2312.08977) | 通过加权集成模型实现了高准确性的持续学习，兼顾可塑性和稳定性。 |
| [^5] | [Image Inpainting via Tractable Steering of Diffusion Models.](http://arxiv.org/abs/2401.03349) | 本文提出了一种通过可解概率模型精确计算约束后验的方法，然后利用这一信号来引导扩散模型的去噪过程，从而改进图像修复的质量和语义一致性。 |
| [^6] | [Trojan Model Detection Using Activation Optimization.](http://arxiv.org/abs/2306.04877) | 本文提出了一种新颖的特洛伊模型检测方法，通过激活优化为模型创建签名，然后训练分类器来检测特洛伊模型。该方法在两个公共数据集上实现了最先进的性能。 |

# 详细

[^1]: 只需转移它：测试时间原型转移用于视觉语言模型的零样本泛化

    Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models

    [https://arxiv.org/abs/2403.12952](https://arxiv.org/abs/2403.12952)

    引入了测试时间原型转移（TPS）框架，通过动态学习每个原型的转移向量，有效地弥合了领域差距并增强了类

    

    视觉语言模型（VLMs）的进展推动了计算机视觉领域的发展，特别是在零样本学习设置中。尽管它们很有前景，但这些模型的有效性在测试环境中往往会因为领域转移而降低。为了解决这个问题，我们引入了测试时间原型转移（TPS）框架，这是一种旨在使用标记测试输入来使VLM适应测试数据集的开创性方法。我们的方法基于在共享嵌入空间中调节每个类别的原型的概念。通过使用预先训练的文本编码器生成并缓存原型，TPS不仅促进了无需优化的原型重用进行后续预测，还让其能够无缝集成当前进展的提示工程技术。在测试时间，TPS仅基于给定的测试样本动态学习每个原型的转移向量，有效地弥合领域差距并增强类

    arXiv:2403.12952v1 Announce Type: cross  Abstract: Advancements in vision-language models (VLMs) have propelled the field of computer vision, particularly in the zero-shot learning setting. Despite their promise, the effectiveness of these models often diminishes due to domain shifts in test environments. To address this, we introduce the Test-Time Prototype Shifting (TPS) framework, a pioneering approach designed to adapt VLMs to test datasets using unlabeled test inputs. Our method is based on the notion of modulating per-class prototypes in the shared embedding space. By pre-computing and caching prototypes generated with the pre-trained text encoder, TPS not only facilitates optimization-free prototype reuse for subsequent predictions but also enables seamless integration with current advancements in prompt engineering. At test-time, TPS dynamically learns shift vectors for each prototype based solely on the given test sample, effectively bridging the domain gap and enhancing class
    
[^2]: 将大型语言模型中的领域特定内容融入知识图谱，以增强零样本对象状态分类

    Fusing Domain-Specific Content from Large Language Models into Knowledge Graphs for Enhanced Zero Shot Object State Classification

    [https://arxiv.org/abs/2403.12151](https://arxiv.org/abs/2403.12151)

    大型语言模型与知识图谱结合，提高零样本对象状态分类性能

    

    领域特定知识可以显著有助于解决各种视觉任务，但生成这种知识需要大量人力和时间成本。本研究探讨了大型语言模型（LLMs）在通过语义嵌入生成和提供领域特定信息方面的潜力。为实现这一目标，将LLM集成到一个流程中，该流程在视觉基础零样本对象状态分类任务的背景下利用知识图谱和预训练的语义向量。通过广泛的消融研究彻底研究了LLM的行为。我们的研究结果表明，将基于LLM的嵌入与通用的预训练嵌入结合使用可以显著提高性能。借鉴这一消融研究的见解，我们对竞争模型进行了比较分析，从而突出了最新的表现水平。

    arXiv:2403.12151v1 Announce Type: new  Abstract: Domain-specific knowledge can significantly contribute to addressing a wide variety of vision tasks. However, the generation of such knowledge entails considerable human labor and time costs. This study investigates the potential of Large Language Models (LLMs) in generating and providing domain-specific information through semantic embeddings. To achieve this, an LLM is integrated into a pipeline that utilizes Knowledge Graphs and pre-trained semantic vectors in the context of the Vision-based Zero-shot Object State Classification task. We thoroughly examine the behavior of the LLM through an extensive ablation study. Our findings reveal that the integration of LLM-based embeddings, in combination with general-purpose pre-trained embeddings, leads to substantial performance improvements. Drawing insights from this ablation study, we conduct a comparative analysis against competing models, thereby highlighting the state-of-the-art perfor
    
[^3]: ChatGPT与生物识别技术：对面部识别、性别检测和年龄估计能力的评估

    ChatGPT and biometrics: an assessment of face recognition, gender detection, and age estimation capabilities

    [https://arxiv.org/abs/2403.02965](https://arxiv.org/abs/2403.02965)

    本文评估了ChatGPT在面部识别、性别检测和年龄估计等生物识别任务中的表现，结果显示ChatGPT在面部识别方面具有较高准确性，并在性别检测方面表现显著，在年龄估计任务中也具有相当准确性。

    

    本文探讨了大型语言模型（LLMs），如ChatGPT，在生物识别任务中的应用。我们特别检验了ChatGPT在执行生物识别相关任务方面的能力，重点关注面部识别、性别检测和年龄估计。由于生物识别被视为敏感信息，ChatGPT避免回答直接提示，因此我们设计了提示策略来绕过其保护措施，并评估生物识别任务的能力。我们的研究表明，ChatGPT能够以相当高的准确性识别面部身份并在两个面部图像之间区分。此外，实验结果显示在性别检测方面性能显著，并对年龄估计任务有相当准确性能。我们的发现揭示了在生物识别中应用LLMs和基础模型具有广阔的潜力。

    arXiv:2403.02965v1 Announce Type: cross  Abstract: This paper explores the application of large language models (LLMs), like ChatGPT, for biometric tasks. We specifically examine the capabilities of ChatGPT in performing biometric-related tasks, with an emphasis on face recognition, gender detection, and age estimation. Since biometrics are considered as sensitive information, ChatGPT avoids answering direct prompts, and thus we crafted a prompting strategy to bypass its safeguard and evaluate the capabilities for biometrics tasks. Our study reveals that ChatGPT recognizes facial identities and differentiates between two facial images with considerable accuracy. Additionally, experimental results demonstrate remarkable performance in gender detection and reasonable accuracy for the age estimation tasks. Our findings shed light on the promising potentials in the application of LLMs and foundation models for biometrics.
    
[^4]: 加权集成模型是强大的持续学习者

    Weighted Ensemble Models Are Strong Continual Learners

    [https://arxiv.org/abs/2312.08977](https://arxiv.org/abs/2312.08977)

    通过加权集成模型实现了高准确性的持续学习，兼顾可塑性和稳定性。

    

    在本文中，我们研究持续学习（CL）的问题，其中目标是从一系列任务中学习模型，使得以前任务的数据在学习当前任务数据时不可用。CL本质上是在能够学习新任务（即可塑性）和保持先前学习概念的性能（即稳定性）之间取得平衡的过程。为了解决稳定性-可塑性的权衡问题，我们建议对先前和当前任务的模型参数进行加权集成。这种加权集成模型，我们称之为持续模型平均（或CoMA），通过利用可塑性在当前任务上获得高准确性，同时不会偏离太远的先前权重配置，从而确保稳定性。我们还提出了CoMA的改进型变体，名为持续费舍尔加权模型平均（或CoFiMA），该模型对每一个参数进行选择性加权。

    arXiv:2312.08977v2 Announce Type: replace-cross  Abstract: In this work, we study the problem of continual learning (CL) where the goal is to learn a model on a sequence of tasks, such that the data from the previous tasks becomes unavailable while learning on the current task data. CL is essentially a balancing act between being able to learn on the new task (i.e., plasticity) and maintaining the performance on the previously learned concepts (i.e., stability). Intending to address the stability-plasticity trade-off, we propose to perform weight-ensembling of the model parameters of the previous and current tasks. This weighted-ensembled model, which we call Continual Model Averaging (or CoMA), attains high accuracy on the current task by leveraging plasticity, while not deviating too far from the previous weight configuration, ensuring stability. We also propose an improved variant of CoMA, named Continual Fisher-weighted Model Averaging (or CoFiMA), that selectively weighs each para
    
[^5]: 图像修复通过可控扩散模型的导航

    Image Inpainting via Tractable Steering of Diffusion Models. (arXiv:2401.03349v1 [cs.CV])

    [http://arxiv.org/abs/2401.03349](http://arxiv.org/abs/2401.03349)

    本文提出了一种通过可解概率模型精确计算约束后验的方法，然后利用这一信号来引导扩散模型的去噪过程，从而改进图像修复的质量和语义一致性。

    

    扩散模型是生成逼真图像的当前最先进技术。然而，对于有约束的图像生成任务，如修复，控制抽样过程仍然具有挑战性，因为对这些约束的精确条件设定是不可解的。本文提出利用可解的概率模型(TPMs)的能力来精确且有效地计算受约束的后验，并利用该信号来引导扩散模型的去噪过程。具体而言，本文采用了一类表达力较强的TPMs，称为概率电路(PCs)。基于先前的进展，我们进一步扩大了PCs的规模，并使其能够引导扩散模型的图像生成过程。实证结果表明，我们的方法可以在三个自然图像数据集（即CelebA-H）中持续改进修复图像的整体质量和语义一致性。

    Diffusion models are the current state of the art for generating photorealistic images. Controlling the sampling process for constrained image generation tasks such as inpainting, however, remains challenging since exact conditioning on such constraints is intractable. While existing methods use various techniques to approximate the constrained posterior, this paper proposes to exploit the ability of Tractable Probabilistic Models (TPMs) to exactly and efficiently compute the constrained posterior, and to leverage this signal to steer the denoising process of diffusion models. Specifically, this paper adopts a class of expressive TPMs termed Probabilistic Circuits (PCs). Building upon prior advances, we further scale up PCs and make them capable of guiding the image generation process of diffusion models. Empirical results suggest that our approach can consistently improve the overall quality and semantic coherence of inpainted images across three natural image datasets (i.e., CelebA-H
    
[^6]: 使用激活优化进行特洛伊模型检测

    Trojan Model Detection Using Activation Optimization. (arXiv:2306.04877v1 [cs.CV])

    [http://arxiv.org/abs/2306.04877](http://arxiv.org/abs/2306.04877)

    本文提出了一种新颖的特洛伊模型检测方法，通过激活优化为模型创建签名，然后训练分类器来检测特洛伊模型。该方法在两个公共数据集上实现了最先进的性能。

    

    由于数据的不可用性或大规模，以及训练机器学习模型的高计算和人力成本，通常会在可能的情况下依赖于开源预训练模型。但是，从安全的角度来看，这种做法非常令人担忧。预训练模型可能会被感染特洛伊攻击，在这种攻击中，攻击者嵌入一个触发器在模型中，使得当触发器存在于输入中时，攻击者可以控制模型的行为。本文提出了一种新颖的特洛伊模型检测方法的初步工作。我们的方法根据激活优化为模型创建签名。然后训练分类器来检测特洛伊模型并给出其签名。我们的方法在两个公共数据集上实现了最先进的性能。

    Due to data's unavailability or large size, and the high computational and human labor costs of training machine learning models, it is a common practice to rely on open source pre-trained models whenever possible. However, this practice is worry some from the security perspective. Pre-trained models can be infected with Trojan attacks, in which the attacker embeds a trigger in the model such that the model's behavior can be controlled by the attacker when the trigger is present in the input. In this paper, we present our preliminary work on a novel method for Trojan model detection. Our method creates a signature for a model based on activation optimization. A classifier is then trained to detect a Trojan model given its signature. Our method achieves state of the art performance on two public datasets.
    

