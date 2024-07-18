# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models](https://arxiv.org/abs/2403.13890) | 提出了一个多条件潜在扩散模型来学习对比动力学，以减少对静脉内对比剂的依赖性。 |
| [^2] | [Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt](https://arxiv.org/abs/2403.09857) | 提出了一个名为ASP的框架，通过注意力方面减少特定信息，鼓励任务不变的提示来捕获共享知识，并通过信息瓶颈学习目标从旧类到新类传递知识。 |
| [^3] | [The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?](https://arxiv.org/abs/2403.09037) | 本研究利用线性探测揭示了大型视觉语言模型的隐藏知识，发现首个令牌的logit分布包含足够信息，可以识别无法回答的视觉问题、防范多模态越狱攻击以及识别欺骗性问题，并提出了一个简单的解码策略以有效改善生成内容。 |
| [^4] | [How we won BraTS 2023 Adult Glioma challenge? Just faking it! Enhanced Synthetic Data Augmentation and Model Ensemble for brain tumour segmentation](https://arxiv.org/abs/2402.17317) | 通过使用生成对抗网络和配准来增强合成数据，我们成功训练了三个不同的深度学习模型，结合卷积算法和transformers技术填补了知识差距，取得了0.9005的dice结果。 |
| [^5] | [Nearest Neighbour Score Estimators for Diffusion Generative Models](https://arxiv.org/abs/2402.08018) | 本论文提出了一种新颖的最近邻评分函数估计器，通过利用训练集中的多个样本大大降低了估计器的方差，可用于训练一致性模型和扩散模型，提高收敛速度、样本质量，并为进一步的研究提供了新的可能性。 |
| [^6] | [Scissorhands: Scrub Data Influence via Connection Sensitivity in Networks.](http://arxiv.org/abs/2401.06187) | Scissorhands 是一种新的机器取消学习方法，通过连接敏感性识别与遗忘数据相关的最相关参数，并通过重新训练修剪的模型来擦除数据影响。 |
| [^7] | [Attribute Based Interpretable Evaluation Metrics for Generative Models.](http://arxiv.org/abs/2310.17261) | 本论文提出了一种基于属性的生成模型可解释性评估指标，通过度量生成图像集与训练集关于属性强度分布的差异，可以更好地衡量模型生成结果与训练数据的相似度。 |
| [^8] | [EGIC: Enhanced Low-Bit-Rate Generative Image Compression Guided by Semantic Segmentation.](http://arxiv.org/abs/2309.03244) | EGIC是一种增强的低位速率生成图像压缩方法，通过语义分割提供指导。它在失真感知和失真方向基线方法上表现优越，并具有较小的模型参数和优秀的插值特性。 |
| [^9] | [Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review.](http://arxiv.org/abs/2308.05731) | 这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。 |
| [^10] | [From Fake to Real (FFR): A two-stage training pipeline for mitigating spurious correlations with synthetic data.](http://arxiv.org/abs/2308.04553) | 本文提出了一个两阶段训练流程，通过在一个平衡的合成数据集上进行预训练，然后在真实数据上进行微调，减少了视觉识别模型学习到与数据集偏差相关的错误的问题。 |
| [^11] | [Spatio-Temporal Branching for Motion Prediction using Motion Increments.](http://arxiv.org/abs/2308.01097) | 本论文提出了一种利用运动增量进行时空分支的运动预测网络，通过解耦时域和空域特征的学习，提取更多的运动信息。 |
| [^12] | [DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks.](http://arxiv.org/abs/2306.09124) | DIFFender是一种基于扩散的对抗性防御方法，通过定位和恢复两个阶段的操作，利用文本引导的扩散模型来防御对抗性Patch，从而提高其整体防御性能。 |
| [^13] | [SSL-Cleanse: Trojan Detection and Mitigation in Self-Supervised Learning.](http://arxiv.org/abs/2303.09079) | 本篇论文讨论了自监督学习中的木马攻击检测和缓解问题。由于这种攻击危险隐匿，且在下游分类器中很难检测出来。目前在超监督学习中的木马检测方法可以潜在地保护SSL下游分类器，但在其广泛传播之前识别和处理SSL编码器中的触发器是一项艰巨的任务。 |
| [^14] | [Similarity of Neural Architectures Based on Input Gradient Transferability.](http://arxiv.org/abs/2210.11407) | 本研究利用对抗攻击传递度量，设计了一个量化且可扩展的神经架构相似度函数，分析了69个最先进的ImageNet分类器，发现多样化的神经架构可以提高模型集合和知识蒸馏的性能。 |

# 详细

[^1]: 以多条件潜在扩散模型学习对比动力学

    Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models

    [https://arxiv.org/abs/2403.13890](https://arxiv.org/abs/2403.13890)

    提出了一个多条件潜在扩散模型来学习对比动力学，以减少对静脉内对比剂的依赖性。

    

    动态对比增强磁共振成像中的对比剂可以定位肿瘤并观察其对比动力学，这对于癌症表征和治疗决策至关重要。然而，对比剂的使用不仅与不良健康风险相关，而且对于怀孕患者、肾功能障碍患者或其他不良反应患者存在限制。由于对比剂摄取是病灶恶性、癌症复发风险和治疗反应的关键生物标志物，因此减少静脉内对比剂的依赖性变得至关重要。为此，我们提出了一个能够进行DCE-MRI时间序列的获取时间条件图像合成的多条件潜在扩散模型。为了评估医学图像合成，我们还提出并验证了基于生物标志物变异性的Fr\'echet放射组学距离作为图像质量度量。

    arXiv:2403.13890v1 Announce Type: cross  Abstract: Contrast agents in dynamic contrast enhanced magnetic resonance imaging allow to localize tumors and observe their contrast kinetics, which is essential for cancer characterization and respective treatment decision-making. However, contrast agent administration is not only associated with adverse health risks, but also restricted for patients during pregnancy, and for those with kidney malfunction, or other adverse reactions. With contrast uptake as key biomarker for lesion malignancy, cancer recurrence risk, and treatment response, it becomes pivotal to reduce the dependency on intravenous contrast agent administration. To this end, we propose a multi-conditional latent diffusion model capable of acquisition time-conditioned image synthesis of DCE-MRI temporal sequences. To evaluate medical image synthesis, we additionally propose and validate the Fr\'echet radiomics distance as an image quality measure based on biomarker variability 
    
[^2]: 带有注意力感知自适应提示的少样本类增量学习

    Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt

    [https://arxiv.org/abs/2403.09857](https://arxiv.org/abs/2403.09857)

    提出了一个名为ASP的框架，通过注意力方面减少特定信息，鼓励任务不变的提示来捕获共享知识，并通过信息瓶颈学习目标从旧类到新类传递知识。

    

    少样本类增量学习（FSCIL）模型旨在在保留旧类知识的同时，逐步学习新类别的稀缺样本。现有的FSCIL方法通常对整个骨干进行微调，导致过拟合并阻碍学习新类别的潜力。另一方面，最近基于提示的CIL方法通过在每个任务中用足够的数据训练提示来减轻遗忘。在这项工作中，我们提出了一个名为注意力感知自适应提示（ASP）的新框架。ASP通过从注意力方面减少特定信息，鼓励任务不变的提示来捕获共享知识。此外，ASP中的自适应任务特定提示提供特定信息，并通过信息瓶颈学习目标从旧类到新类传递知识。总之，ASP防止了在基础任务上的过拟合，并不需要在少样本增量任务中使用大量数据。

    arXiv:2403.09857v1 Announce Type: cross  Abstract: Few-Shot Class-Incremental Learning (FSCIL) models aim to incrementally learn new classes with scarce samples while preserving knowledge of old ones. Existing FSCIL methods usually fine-tune the entire backbone, leading to overfitting and hindering the potential to learn new classes. On the other hand, recent prompt-based CIL approaches alleviate forgetting by training prompts with sufficient data in each task. In this work, we propose a novel framework named Attention-aware Self-adaptive Prompt (ASP). ASP encourages task-invariant prompts to capture shared knowledge by reducing specific information from the attention aspect. Additionally, self-adaptive task-specific prompts in ASP provide specific information and transfer knowledge from old classes to new classes with an Information Bottleneck learning objective. In summary, ASP prevents overfitting on base task and does not require enormous data in few-shot incremental tasks. Extensi
    
[^3]: 第一个知道：令牌分布如何揭示大型视觉语言模型中的隐藏知识？

    The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?

    [https://arxiv.org/abs/2403.09037](https://arxiv.org/abs/2403.09037)

    本研究利用线性探测揭示了大型视觉语言模型的隐藏知识，发现首个令牌的logit分布包含足够信息，可以识别无法回答的视觉问题、防范多模态越狱攻击以及识别欺骗性问题，并提出了一个简单的解码策略以有效改善生成内容。

    

    大型视觉语言模型（LVLMs）旨在解释和响应人类指令，但由于不当指令而偶尔生成幻觉或有害内容。本研究使用线性探测来揭示LVLMs输出层的隐藏知识。我们证明了首个令牌的logit分布包含足够信息，可以确定是否应对指令作出响应，包括识别无法回答的视觉问题、防范多模态越狱攻击以及识别欺骗性问题。这种隐藏知识在响应生成过程中随后令牌的logit逐渐丢失。然后，我们演示了一种简单的解码策略在生成第一个令牌时，有效改善生成的内容。在实验中，我们发现了一些有趣的见解：首先，CLIP模型已经包含解决这些任务的强烈信号，表明潜力

    arXiv:2403.09037v1 Announce Type: cross  Abstract: Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layer of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against multi-modal jailbreaking attack, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, indicating poten
    
[^4]: 如何赢得BraTS 2023成年胶质瘤挑战？假装而已！增强的合成数据增强和模型集成用于脑肿瘤分割

    How we won BraTS 2023 Adult Glioma challenge? Just faking it! Enhanced Synthetic Data Augmentation and Model Ensemble for brain tumour segmentation

    [https://arxiv.org/abs/2402.17317](https://arxiv.org/abs/2402.17317)

    通过使用生成对抗网络和配准来增强合成数据，我们成功训练了三个不同的深度学习模型，结合卷积算法和transformers技术填补了知识差距，取得了0.9005的dice结果。

    

    深度学习是颅内肿瘤分割的最先进技术，但这需要大量高质量数据，尤其在医学领域难以获得。因此，我们的解决方案通过使用非传统的数据增强机制来解决这个问题。生成对抗网络和配准被用来大量增加可用样本数，用于训练三个不同的深度学习模型，分别用于颅内肿瘤分割的BraTS2023挑战的第一个任务。第一个模型是标准nnU-Net，第二个是Swin UNETR，第三个是BraTS 2021挑战的获胜方案。整个流程基于nnU-Net实现，除了合成数据的生成。卷积算法和transformers的使用能够填补彼此的知识差距。使用新指标，我们的最佳解决方案达到了0.9005的dice结果。

    arXiv:2402.17317v1 Announce Type: cross  Abstract: Deep Learning is the state-of-the-art technology for segmenting brain tumours. However, this requires a lot of high-quality data, which is difficult to obtain, especially in the medical field. Therefore, our solutions address this problem by using unconventional mechanisms for data augmentation. Generative adversarial networks and registration are used to massively increase the amount of available samples for training three different deep learning models for brain tumour segmentation, the first task of the BraTS2023 challenge. The first model is the standard nnU-Net, the second is the Swin UNETR and the third is the winning solution of the BraTS 2021 Challenge. The entire pipeline is built on the nnU-Net implementation, except for the generation of the synthetic data. The use of convolutional algorithms and transformers is able to fill each other's knowledge gaps. Using the new metric, our best solution achieves the dice results 0.9005
    
[^5]: 扩散生成模型的最近邻评分估计器

    Nearest Neighbour Score Estimators for Diffusion Generative Models

    [https://arxiv.org/abs/2402.08018](https://arxiv.org/abs/2402.08018)

    本论文提出了一种新颖的最近邻评分函数估计器，通过利用训练集中的多个样本大大降低了估计器的方差，可用于训练一致性模型和扩散模型，提高收敛速度、样本质量，并为进一步的研究提供了新的可能性。

    

    评分函数估计是训练和采样扩散生成模型的基础。尽管如此，最常用的估计器要么是有偏的神经网络逼近，要么是基于条件评分的高方差蒙特卡洛估计器。我们引入了一种创新的最近邻评分函数估计器，利用训练集中的多个样本大大降低了估计器的方差。我们在两个引人注目的应用中利用了低方差估计器。在使用我们的估计器进行训练一致性模型时，我们报告了收敛速度和样本质量显著提高。在扩散模型中，我们展示了我们的估计器可以替代学习网络进行概率流ODE积分，为未来研究开辟了有前景的新方向。

    Score function estimation is the cornerstone of both training and sampling from diffusion generative models. Despite this fact, the most commonly used estimators are either biased neural network approximations or high variance Monte Carlo estimators based on the conditional score. We introduce a novel nearest neighbour score function estimator which utilizes multiple samples from the training set to dramatically decrease estimator variance. We leverage our low variance estimator in two compelling applications. Training consistency models with our estimator, we report a significant increase in both convergence speed and sample quality. In diffusion models, we show that our estimator can replace a learned network for probability-flow ODE integration, opening promising new avenues of future research.
    
[^6]: Scissorhands: 通过网络连接敏感性在数据影响中进行数据擦除

    Scissorhands: Scrub Data Influence via Connection Sensitivity in Networks. (arXiv:2401.06187v1 [cs.LG])

    [http://arxiv.org/abs/2401.06187](http://arxiv.org/abs/2401.06187)

    Scissorhands 是一种新的机器取消学习方法，通过连接敏感性识别与遗忘数据相关的最相关参数，并通过重新训练修剪的模型来擦除数据影响。

    

    机器取消学习已成为一项重要任务，旨在擦除训练模型中的数据影响。它符合最新的数据监管标准，增强了机器学习应用的隐私和安全性。大多数现有的机器取消学习方法表现良好，但通常需要访问其余数据的全部内容，在某些情况下可能不可行。在这项工作中，我们提出了一种新的机器取消学习方法“Scissorhands”，它只使用训练数据的子集来有效运行。初始阶段，Scissorhands通过连接敏感性在给定模型中识别与遗忘数据相关的最相关参数。该过程通过重新初始化这些参数中具有最大影响力的前k%的最相关参数，从而产生一个用于擦除遗忘数据影响的修剪模型。随后，Scissorhands通过最小-最大优化过程对修剪的模型进行再训练，寻找保留信息的参数。

    Machine unlearning has become a pivotal task to erase the influence of data from a trained model. It adheres to recent data regulation standards and enhances the privacy and security of machine learning applications. Most existing machine unlearning methods perform well, however, they typically necessitate access to the entirety of the remaining data, which might not be feasible in certain scenarios. In this work, we present a new machine unlearning approach Scissorhands, which operates effectively with only a subset of the training data. Initially, Scissorhands identifies the most pertinent parameters in the given model relative to the forgetting data via connection sensitivity. This process involves reinitializing the most influential top-$k$ percent of these parameters, resulting in a trimmed model for erasing the influence of the forgetting data. Subsequently, Scissorhands retrains the trimmed model through a min-max optimization process, seeking parameters that preserve informatio
    
[^7]: 基于属性的生成模型可解释性评估指标

    Attribute Based Interpretable Evaluation Metrics for Generative Models. (arXiv:2310.17261v1 [cs.CV])

    [http://arxiv.org/abs/2310.17261](http://arxiv.org/abs/2310.17261)

    本论文提出了一种基于属性的生成模型可解释性评估指标，通过度量生成图像集与训练集关于属性强度分布的差异，可以更好地衡量模型生成结果与训练数据的相似度。

    

    当训练数据集中狗和猫的比例为1:1时，生成模型生成的狗和猫也应更好地符合训练数据集的分布。然而，现有的评估指标只提供了“多样性”这个解释性之外的维度。在这个背景下，我们提出了一种新的评估协议，通过度量生成图像集与训练集关于属性强度分布的差异来捕捉这种现象。单属性差异（SaD）衡量了关于单个属性的概率密度函数的差异。双属性差异（PaD）衡量了关于一对属性的联合概率密度函数的差异。它们提供了模型所面临的困难属性。为了衡量图像的属性强度，我们提出了异构CLIP评分（HCS），它通过测量图像和文本向量之间的余弦相似度来实现。

    When the training dataset comprises a 1:1 proportion of dogs to cats, a generative model that produces 1:1 dogs and cats better resembles the training species distribution than another model with 3:1 dogs and cats. Can we capture this phenomenon using existing metrics? Unfortunately, we cannot, because these metrics do not provide any interpretability beyond "diversity". In this context, we propose a new evaluation protocol that measures the divergence of a set of generated images from the training set regarding the distribution of attribute strengths as follows. Single-attribute Divergence (SaD) measures the divergence regarding PDFs of a single attribute. Paired-attribute Divergence (PaD) measures the divergence regarding joint PDFs of a pair of attributes. They provide which attributes the models struggle. For measuring the attribute strengths of an image, we propose Heterogeneous CLIPScore (HCS) which measures the cosine similarity between image and text vectors with heterogeneous 
    
[^8]: EGIC:增强的低位速率生成图像压缩方法在语义分割的指导下

    EGIC: Enhanced Low-Bit-Rate Generative Image Compression Guided by Semantic Segmentation. (arXiv:2309.03244v1 [eess.IV])

    [http://arxiv.org/abs/2309.03244](http://arxiv.org/abs/2309.03244)

    EGIC是一种增强的低位速率生成图像压缩方法，通过语义分割提供指导。它在失真感知和失真方向基线方法上表现优越，并具有较小的模型参数和优秀的插值特性。

    

    我们引入了一种新颖的生成图像压缩方法EGIC，它允许从一个单一模型有效地遍历失真感知曲线。具体而言，我们提出了一种隐式编码的图像插值变体，用于预测在MSE优化和GAN优化解码器输出之间的残差。在接收端，用户可以控制残差对基于GAN的重建的影响。结合改进的基于GAN的构建块，EGIC在感知导向和失真导向的基线方法（包括HiFiC，MRIC和DIRAC）上表现优于大多数方法，在失真端与VTM-20.0几乎相当。EGIC实现简单，非常轻量级（与HiFiC相比，模型参数只有0.18倍），并提供优异的插值特性，这使得它成为针对低位范围的实际应用的有希望的候选方法。

    We introduce EGIC, a novel generative image compression method that allows traversing the distortion-perception curve efficiently from a single model. Specifically, we propose an implicitly encoded variant of image interpolation that predicts the residual between a MSE-optimized and GAN-optimized decoder output. On the receiver side, the user can then control the impact of the residual on the GAN-based reconstruction. Together with improved GAN-based building blocks, EGIC outperforms a wide-variety of perception-oriented and distortion-oriented baselines, including HiFiC, MRIC and DIRAC, while performing almost on par with VTM-20.0 on the distortion end. EGIC is simple to implement, very lightweight (e.g. 0.18x model parameters compared to HiFiC) and provides excellent interpolation characteristics, which makes it a promising candidate for practical applications targeting the low bit range.
    
[^9]: 重新思考基于深度学习的自动驾驶系统中的预测和规划的整合：一项综述

    Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review. (arXiv:2308.05731v1 [cs.RO])

    [http://arxiv.org/abs/2308.05731](http://arxiv.org/abs/2308.05731)

    这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。

    

    自动驾驶有可能彻底改变个人、公共和货运交通的方式。除了感知环境的巨大挑战外，即准确地使用可用的传感器数据感知环境，自动驾驶还包括规划一个安全、舒适和高效的运动轨迹。为了促进安全和进步，许多工作依赖于模块化的交通未来运动的预测。模块化的自动驾驶系统通常将预测和规划作为顺序的独立任务处理。虽然这考虑了周围交通对自车的影响，但它未能预测交通参与者对自车行为的反应。最近的研究表明，将预测和规划整合为相互依赖的联合步骤是实现安全、高效和舒适驾驶所必需的。虽然有各种模型实现了这种集成系统，但对不同原理的全面概述和理论理解仍然缺乏。

    Automated driving has the potential to revolutionize personal, public, and freight mobility. Besides the enormous challenge of perception, i.e. accurately perceiving the environment using available sensor data, automated driving comprises planning a safe, comfortable, and efficient motion trajectory. To promote safety and progress, many works rely on modules that predict the future motion of surrounding traffic. Modular automated driving systems commonly handle prediction and planning as sequential separate tasks. While this accounts for the influence of surrounding traffic on the ego-vehicle, it fails to anticipate the reactions of traffic participants to the ego-vehicle's behavior. Recent works suggest that integrating prediction and planning in an interdependent joint step is necessary to achieve safe, efficient, and comfortable driving. While various models implement such integrated systems, a comprehensive overview and theoretical understanding of different principles are lacking.
    
[^10]: 从假到真（FFR）：一种用于减少与合成数据相关性错误的两阶段训练流程

    From Fake to Real (FFR): A two-stage training pipeline for mitigating spurious correlations with synthetic data. (arXiv:2308.04553v1 [cs.CV])

    [http://arxiv.org/abs/2308.04553](http://arxiv.org/abs/2308.04553)

    本文提出了一个两阶段训练流程，通过在一个平衡的合成数据集上进行预训练，然后在真实数据上进行微调，减少了视觉识别模型学习到与数据集偏差相关的错误的问题。

    

    视觉识别模型容易学习到由于训练集的不平衡导致的相关性错误，其中某些群体（如女性）在某些类别（如程序员）中代表性不足。生成模型通过为少数样本生成合成数据来减少这种偏差，从而平衡训练集。然而，先前使用这些方法的工作忽视了视觉识别模型往往能够学习区分真实图像和合成图像的能力，因此无法消除原始数据集中的偏差。在我们的工作中，我们提出了一种新颖的两阶段流程来减少这个问题，其中1）我们在平衡的合成数据集上进行预训练，然后2）在真实数据上进行微调。使用这个流程，我们避免了在真实数据和合成数据上的训练，从而避免了真实数据和合成数据之间的偏差。此外，在第一步中我们学习到了抵抗偏差的稳健特征，在第二步中减轻了偏差。

    Visual recognition models are prone to learning spurious correlations induced by an imbalanced training set where certain groups (\eg Females) are under-represented in certain classes (\eg Programmers). Generative models offer a promising direction in mitigating this bias by generating synthetic data for the minority samples and thus balancing the training set. However, prior work that uses these approaches overlooks that visual recognition models could often learn to differentiate between real and synthetic images and thus fail to unlearn the bias in the original dataset. In our work, we propose a novel two-stage pipeline to mitigate this issue where 1) we pre-train a model on a balanced synthetic dataset and then 2) fine-tune on the real data. Using this pipeline, we avoid training on both real and synthetic data, thus avoiding the bias between real and synthetic data. Moreover, we learn robust features against the bias in the first step that mitigate the bias in the second step. Mor
    
[^11]: 利用运动增量进行时空分支的运动预测

    Spatio-Temporal Branching for Motion Prediction using Motion Increments. (arXiv:2308.01097v1 [cs.CV])

    [http://arxiv.org/abs/2308.01097](http://arxiv.org/abs/2308.01097)

    本论文提出了一种利用运动增量进行时空分支的运动预测网络，通过解耦时域和空域特征的学习，提取更多的运动信息。

    

    人体运动预测已成为一个热门的研究课题，但由于未来姿势的随机和不规则性质，这仍然是一个具有挑战性的任务。传统方法依赖于手工特征和机器学习技术，往往难以建模人体运动的复杂动力学。最近基于深度学习的方法通过学习运动的时空表示取得了成功，但这些模型常常忽视运动数据的可靠性。此外，骨架节点的时域和空域依赖性是不同的。时域关系捕捉到随时间的运动信息，而空域关系描述了身体结构和不同节点之间的关系。在本文中，我们提出了一种新颖的利用增量信息进行时空分支的运动预测网络，它解耦了时域和空域特征的学习，提取了更多的运动信息。

    Human motion prediction (HMP) has emerged as a popular research topic due to its diverse applications, but it remains a challenging task due to the stochastic and aperiodic nature of future poses. Traditional methods rely on hand-crafted features and machine learning techniques, which often struggle to model the complex dynamics of human motion. Recent deep learning-based methods have achieved success by learning spatio-temporal representations of motion, but these models often overlook the reliability of motion data. Additionally, the temporal and spatial dependencies of skeleton nodes are distinct. The temporal relationship captures motion information over time, while the spatial relationship describes body structure and the relationships between different nodes. In this paper, we propose a novel spatio-temporal branching network using incremental information for HMP, which decouples the learning of temporal-domain and spatial-domain features, extracts more motion information, and ac
    
[^12]: DIFFender：基于扩散的对抗性防御方法用于抵御Patch攻击

    DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks. (arXiv:2306.09124v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.09124](http://arxiv.org/abs/2306.09124)

    DIFFender是一种基于扩散的对抗性防御方法，通过定位和恢复两个阶段的操作，利用文本引导的扩散模型来防御对抗性Patch，从而提高其整体防御性能。

    

    对抗性攻击，尤其是Patch攻击，对深度学习模型的鲁棒性和可靠性构成了重大威胁。开发可靠的防御方法以抵御Patch攻击对于实际应用至关重要，然而当前在这个领域的研究还不令人满意。在本文中，我们提出了DIFFender，一种新颖的防御方法，它利用文本引导的扩散模型来防御对抗性Patch。DIFFender包括两个主要阶段：Patch定位和Patch恢复。在定位阶段，我们发现并利用了扩散模型的一个有趣特性，以有效地识别对抗性Patch的位置。在恢复阶段，我们利用扩散模型重建图像中的对抗性区域同时保持视觉内容的完整性。重要的是，这两个阶段都受到统一的扩散模型的精心引导，因此我们可以利用它们之间的紧密相互作用来提高整个防御性能。

    Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is not satisfactory. In this paper, we propose DIFFender, a novel defense method that leverages a text-guided diffusion model to defend against adversarial patches. DIFFender includes two main stages: patch localization and patch restoration. In the localization stage, we find and exploit an intriguing property of the diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ the diffusion model to reconstruct the adversarial regions in the images while preserving the integrity of the visual content. Importantly, these two stages are carefully guided by a unified diffusion model, thus we can utilize the close interaction between them to improve the whole defense performance. Mor
    
[^13]: SSL清理：自监督学习中的木马检测和缓解

    SSL-Cleanse: Trojan Detection and Mitigation in Self-Supervised Learning. (arXiv:2303.09079v1 [cs.CR])

    [http://arxiv.org/abs/2303.09079](http://arxiv.org/abs/2303.09079)

    本篇论文讨论了自监督学习中的木马攻击检测和缓解问题。由于这种攻击危险隐匿，且在下游分类器中很难检测出来。目前在超监督学习中的木马检测方法可以潜在地保护SSL下游分类器，但在其广泛传播之前识别和处理SSL编码器中的触发器是一项艰巨的任务。

    

    自监督学习（SSL）是一种常用的学习和编码数据表示的方法。通过使用预先训练的SSL图像编码器并在其顶部训练下游分类器，可以在各种任务上实现令人印象深刻的性能，而只需很少的标记数据。SSL的增加使用导致了与SSL编码器相关的安全研究和各种木马攻击的发展。在SSL编码器中插入木马攻击的危险在于它们能够隐蔽地操作并在各种用户和设备之间广泛传播。Trojaned编码器中的后门行为的存在可能会被下游分类器意外继承，使检测和缓解威胁变得更加困难。虽然超监督学习中当前的木马检测方法可以潜在地保护SSL下游分类器，但在其广泛传播之前识别和处理SSL编码器中的触发器是一项艰巨的任务。

    Self-supervised learning (SSL) is a commonly used approach to learning and encoding data representations. By using a pre-trained SSL image encoder and training a downstream classifier on top of it, impressive performance can be achieved on various tasks with very little labeled data. The increasing usage of SSL has led to an uptick in security research related to SSL encoders and the development of various Trojan attacks. The danger posed by Trojan attacks inserted in SSL encoders lies in their ability to operate covertly and spread widely among various users and devices. The presence of backdoor behavior in Trojaned encoders can inadvertently be inherited by downstream classifiers, making it even more difficult to detect and mitigate the threat. Although current Trojan detection methods in supervised learning can potentially safeguard SSL downstream classifiers, identifying and addressing triggers in the SSL encoder before its widespread dissemination is a challenging task. This is be
    
[^14]: 基于输入梯度传递的神经架构相似性研究

    Similarity of Neural Architectures Based on Input Gradient Transferability. (arXiv:2210.11407v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.11407](http://arxiv.org/abs/2210.11407)

    本研究利用对抗攻击传递度量，设计了一个量化且可扩展的神经架构相似度函数，分析了69个最先进的ImageNet分类器，发现多样化的神经架构可以提高模型集合和知识蒸馏的性能。

    

    近年来，为图像分类而开发了大量的深度神经架构，这些模型是否相似或不同，以及什么因素影响它们的相似性或不同尚未得到充分的研究。本文旨在设计一个量化且可扩展的神经架构相似度函数以回答这个问题。我们利用对抗攻击传递度量，该度量具有与输入梯度和决策边界相关的信息，被广泛用于理解模型行为。我们使用所提出的相似度函数对69个最先进的ImageNet分类器进行了大规模分析，从而回答了这个问题。此外，我们观察到与神经架构相关的现象，即模型多样性可以在特定条件下对模型集合和知识蒸馏的性能有所提升。我们的结果为为什么开发具有不同组件的多样化神经架构是必要的提供了见解。

    In recent years, a huge amount of deep neural architectures have been developed for image classification. It remains curious whether these models are similar or different and what factors contribute to their similarities or differences. To address this question, we aim to design a quantitative and scalable similarity function between neural architectures. We utilize adversarial attack transferability, which has information related to input gradients and decision boundaries that are widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why the development of diverse neural architectures with distinct components is necessar
    

