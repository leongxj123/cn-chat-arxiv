# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-centric Prediction Explanation via Kernelized Stein Discrepancy](https://arxiv.org/abs/2403.15576) | 该论文提出了一种基于内核化斯坦不相容性的数据中心预测解释方法，通过利用内核函数识别提供最佳预测支持给测试点的训练样本，取得了优异性能。 |
| [^2] | [Controlled Training Data Generation with Diffusion Models](https://arxiv.org/abs/2403.15309) | 提出了一种使用扩散模型生成控制训练数据的方法，通过两个反馈机制，一方面使用监督模型反馈找到对抗性提示词实现图像生成，另一方面通过引导使生成过程朝向特定目标分布。 |
| [^3] | [Bridge the Modality and Capacity Gaps in Vision-Language Model Selection](https://arxiv.org/abs/2403.13797) | 本文分析了在语言-Only VLM选择中的两个固有挑战：「模态差距」和「能力差距」，并提出了VLM选择中弥合这两个差距的方法 |
| [^4] | [Unlocking the Potential of Multimodal Unified Discrete Representation through Training-Free Codebook Optimization and Hierarchical Alignment](https://arxiv.org/abs/2403.05168) | 通过无需训练的码本优化和分层对齐，本研究提出了一种方法扩展了多模态统一表示的细粒度，并实现了更好的跨模态泛化。 |
| [^5] | [Scalable Density-based Clustering with Random Projections](https://arxiv.org/abs/2402.15679) | sDBSCAN是一种利用随机投影进行高维密度聚类的算法，在速度和准确性上显著优于其他算法。 |
| [^6] | [SVQ: Sparse Vector Quantization for Spatiotemporal Forecasting](https://arxiv.org/abs/2312.03406) | SVQ是一种利用稀疏回归实现简明表示的稀疏向量量化技术，通过保留关键细节和滤除噪声来提高时空预测性能。在实验中，SVQ在五个空间-时间基准数据集上取得了最先进的结果。 |
| [^7] | [Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework.](http://arxiv.org/abs/2306.07992) | 本文提出了一种对抗图像重构及检测框架来保护视觉感知推荐系统，能够防御局部扰动为特征的对抗攻击并且能够在干净和对抗性图像上进行训练来检测对抗性图像。 |
| [^8] | [Differentially Private Synthetic Data via Foundation Model APIs 1: Images.](http://arxiv.org/abs/2305.15560) | 该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。 |
| [^9] | [AUTO: Adaptive Outlier Optimization for Online Test-Time OOD Detection.](http://arxiv.org/abs/2303.12267) | 本文提出了一个称为AUTO的方法，在在线测试时利用未标记的在线数据直接提高OOD检测性能。该方法自适应地优化网络参数并在线检测OOD样本，取得了优于现有方法的结果。 |
| [^10] | [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models.](http://arxiv.org/abs/2211.01095) | DPM-Solver++是一种快速求解器，在扩散概率模型的引导采样中表现出色，可加快样本生成速度。 |

# 详细

[^1]: 基于内核化斯坦不相容性的数据中心预测解释

    Data-centric Prediction Explanation via Kernelized Stein Discrepancy

    [https://arxiv.org/abs/2403.15576](https://arxiv.org/abs/2403.15576)

    该论文提出了一种基于内核化斯坦不相容性的数据中心预测解释方法，通过利用内核函数识别提供最佳预测支持给测试点的训练样本，取得了优异性能。

    

    现有的基于示例的预测解释方法通常通过模型的参数或潜在表示来连接测试和训练数据点。尽管这些方法提供了有关模型预测原因的线索，但它们经常表现出固有的缺陷，比如产生显着的计算开销或生成粗粒度的解释。本文提出了一种高精度和数据中心的解释（HD-Explain），这是一种利用内核化斯坦不相容性（KSD）属性的简单预测解释方法。具体来说，KSD唯一地为经过训练的模型定义了一个参数化的内核函数，用于编码与模型相关的数据相关性。通过利用内核函数，可以有效地识别提供最佳预测支持给测试点的训练样本。我们在多个分类领域进行了彻底的分析和实验，结果表明HD-Explain取得了优异的性能。

    arXiv:2403.15576v1 Announce Type: new  Abstract: Existing example-based prediction explanation methods often bridge test and training data points through the model's parameters or latent representations. While these methods offer clues to the causes of model predictions, they often exhibit innate shortcomings, such as incurring significant computational overhead or producing coarse-grained explanations. This paper presents a Highly-precise and Data-centric Explanation (HD-Explain), a straightforward prediction explanation method exploiting properties of Kernelized Stein Discrepancy (KSD). Specifically, the KSD uniquely defines a parameterized kernel function for a trained model that encodes model-dependent data correlation. By leveraging the kernel function, one can identify training samples that provide the best predictive support to a test point efficiently. We conducted thorough analyses and experiments across multiple classification domains, where we show that HD-Explain outperform
    
[^2]: 使用扩散模型生成控制训练数据

    Controlled Training Data Generation with Diffusion Models

    [https://arxiv.org/abs/2403.15309](https://arxiv.org/abs/2403.15309)

    提出了一种使用扩散模型生成控制训练数据的方法，通过两个反馈机制，一方面使用监督模型反馈找到对抗性提示词实现图像生成，另一方面通过引导使生成过程朝向特定目标分布。

    

    在这项工作中，我们提出了一种方法，可以控制文本到图像生成模型以生成训练数据，专门用于监督学习。与之前那些采用开环方法并预先定义提示词来使用语言模型或人类专业知识生成新数据的作品不同，我们开发了一种自动闭环系统，其中包括两个反馈机制。第一个机制使用来自给定监督模型的反馈，并找到导致图像生成最大化模型损失的对抗提示词。虽然这些对抗提示词导致了经过模型训练的多样化数据生成，但它们并不知道目标分布，这可能效率低下。因此，我们引入第二个反馈机制，将生成过程引导到特定目标分布。我们称将这两个机制结合起来的方法为引导对抗提示词。我们在不同任务上进行评估。

    arXiv:2403.15309v1 Announce Type: cross  Abstract: In this work, we present a method to control a text-to-image generative model to produce training data specifically "useful" for supervised learning. Unlike previous works that employ an open-loop approach and pre-define prompts to generate new data using either a language model or human expertise, we develop an automated closed-loop system which involves two feedback mechanisms. The first mechanism uses feedback from a given supervised model and finds adversarial prompts that result in image generations that maximize the model loss. While these adversarial prompts result in diverse data informed by the model, they are not informed of the target distribution, which can be inefficient. Therefore, we introduce the second feedback mechanism that guides the generation process towards a certain target distribution. We call the method combining these two mechanisms Guided Adversarial Prompts. We perform our evaluations on different tasks, da
    
[^3]: 弥合视觉-语言模型选择中的模态差距和能力差距

    Bridge the Modality and Capacity Gaps in Vision-Language Model Selection

    [https://arxiv.org/abs/2403.13797](https://arxiv.org/abs/2403.13797)

    本文分析了在语言-Only VLM选择中的两个固有挑战：「模态差距」和「能力差距」，并提出了VLM选择中弥合这两个差距的方法

    

    视觉语言模型（VLMs）通过将图像与文本类别名称配对，在零样本图像分类方面表现出色。预训练的VLMs的不断增加使得特定任务的VLM选择更有可能标识出适合的VLM。因此，一种有前途的零样本图像分类策略是从VLM动物园中选择最合适的预训练VLM，仅依赖目标数据集的文本数据而无需访问数据集的图像。本文分析了这种仅语言VLM选择中两个固有挑战：「模态差距」——VLM在两个不同模态下的嵌入之间的差异，使得文本成为图像的一个不太可靠的替代品；「能力差距」——VLM的整体排名与其在目标数据集的排名之间存在差异，阻碍了直接从模型的整体表现来预测其数据集特定性能。我们提出了VLM选择

    arXiv:2403.13797v1 Announce Type: new  Abstract: Vision Language Models (VLMs) excel in zero-shot image classification by pairing images with textual category names. The expanding variety of Pre-Trained VLMs enhances the likelihood of identifying a suitable VLM for specific tasks. Thus, a promising zero-shot image classification strategy is selecting the most appropriate Pre-Trained VLM from the VLM Zoo, relying solely on the text data of the target dataset without access to the dataset's images. In this paper, we analyze two inherent challenges in assessing the ability of a VLM in this Language-Only VLM selection: the "Modality Gap" -- the disparity in VLM's embeddings across two different modalities, making text a less reliable substitute for images; and the "Capability Gap" -- the discrepancy between the VLM's overall ranking and its ranking for target dataset, hindering direct prediction of a model's dataset-specific performance from its general performance. We propose VLM Selectio
    
[^4]: 通过无需训练的码本优化和分层对齐解锁多模态统一离散表示的潜力

    Unlocking the Potential of Multimodal Unified Discrete Representation through Training-Free Codebook Optimization and Hierarchical Alignment

    [https://arxiv.org/abs/2403.05168](https://arxiv.org/abs/2403.05168)

    通过无需训练的码本优化和分层对齐，本研究提出了一种方法扩展了多模态统一表示的细粒度，并实现了更好的跨模态泛化。

    

    最近在表示学习方面的进展表明多模态对齐的重要性。利用统一码本的双交叉模态信息解缠（DCID）模型在实现细粒度表示和跨模态泛化方面取得了令人期待的结果。然而，它仍受到对所有通道的均等对待以及忽视次要事件信息的阻碍，导致来自无关通道的干扰并在细粒度任务中表现有限。因此，在这项工作中，我们提出了一种无需训练的码本优化（TOC）方法，通过在统一空间中选择重要通道来增强模型性能。此外，我们引入了分层双交叉模态信息解缠（H-DCID）方法将信息分离和对齐扩展到两个级别，捕捉更多跨模态细节。实验结果表明显著的改进。

    arXiv:2403.05168v1 Announce Type: cross  Abstract: Recent advances in representation learning have demonstrated the significance of multimodal alignment. The Dual Cross-modal Information Disentanglement (DCID) model, utilizing a unified codebook, shows promising results in achieving fine-grained representation and cross-modal generalization. However, it is still hindered by equal treatment of all channels and neglect of minor event information, resulting in interference from irrelevant channels and limited performance in fine-grained tasks. Thus, in this work, We propose a Training-free Optimization of Codebook (TOC) method to enhance model performance by selecting important channels in the unified space without retraining. Additionally, we introduce the Hierarchical Dual Cross-modal Information Disentanglement (H-DCID) approach to extend information separation and alignment to two levels, capturing more cross-modal details. The experiment results demonstrate significant improvements a
    
[^5]: 使用随机投影的可扩展密度聚类

    Scalable Density-based Clustering with Random Projections

    [https://arxiv.org/abs/2402.15679](https://arxiv.org/abs/2402.15679)

    sDBSCAN是一种利用随机投影进行高维密度聚类的算法，在速度和准确性上显著优于其他算法。

    

    我们提出了一种名为sDBSCAN的算法，在高维空间中使用余弦距离进行可扩展的密度聚类。通过利用随机投影的保邻特性，sDBSCAN能够快速识别核心点及其邻域，这是密度聚类的主要障碍。在理论上，sDBSCAN在较轻的条件下以高概率输出类似于DBSCAN的聚类结构。为了进一步促进sDBSCAN，我们提出了sOPTICS，这是一种用于交互式探索内在聚类结构的可扩展OPTICS。我们还通过随机核特征将sDBSCAN和sOPTICS扩展到L2、L1、$\chi^2$和Jensen-Shannon距离。在实证方面，sDBSCAN在真实世界的百万数据集上比许多其他聚类算法显著更快，并提供更高的准确性。在这些数据集上，sDBSCAN和sOPTICS在几分钟内运行，而scikit-learn的对应算法需要数小时或由于内存不足而无法运行。

    arXiv:2402.15679v1 Announce Type: new  Abstract: We present sDBSCAN, a scalable density-based clustering algorithm in high dimensions with cosine distance. Utilizing the neighborhood-preserving property of random projections, sDBSCAN can quickly identify core points and their neighborhoods, the primary hurdle of density-based clustering. Theoretically, sDBSCAN outputs a clustering structure similar to DBSCAN under mild conditions with high probability. To further facilitate sDBSCAN, we present sOPTICS, a scalable OPTICS for interactive exploration of the intrinsic clustering structure. We also extend sDBSCAN and sOPTICS to L2, L1, $\chi^2$, and Jensen-Shannon distances via random kernel features. Empirically, sDBSCAN is significantly faster and provides higher accuracy than many other clustering algorithms on real-world million-point data sets. On these data sets, sDBSCAN and sOPTICS run in a few minutes, while the scikit-learn's counterparts demand several hours or cannot run due to m
    
[^6]: SVQ: 稀疏向量量化用于时空预测

    SVQ: Sparse Vector Quantization for Spatiotemporal Forecasting

    [https://arxiv.org/abs/2312.03406](https://arxiv.org/abs/2312.03406)

    SVQ是一种利用稀疏回归实现简明表示的稀疏向量量化技术，通过保留关键细节和滤除噪声来提高时空预测性能。在实验中，SVQ在五个空间-时间基准数据集上取得了最先进的结果。

    

    时空预测在许多领域中都是关键，取得好的预测结果需要找到微妙的模式并滤除噪声。为了解决这个问题，我们介绍了稀疏回归向量量化（SVQ）这一新技术，它利用稀疏回归来实现简明表示，这一方法在理论和实践上都比传统的基于聚类的向量量化方法更有优势。这种方法使用回归模型保留原始向量的关键细节，同时通过稀疏设计滤除噪声。此外，我们使用两层MLP和一个广泛的码本来近似稀疏回归过程。这种方法不仅大大降低了计算成本，还使得SVQ具有可微性和训练简易性，从而显著提高了性能。我们在五个空间-时间基准数据集上进行的实证研究表明，SVQ取得了最先进的结果。具体来说，在

    Spatio-temporal forecasting, pivotal in numerous fields, hinges on the delicate equilibrium between isolating nuanced patterns and sifting out noise. To tackle this, we introduce Sparse Regression-based Vector Quantization (SVQ), a novel technique that leverages sparse regression for succinct representation, an approach theoretically and practically favored over classical clustering-based vector quantization methods. This approach preserves critical details from the original vectors using a regression model while filtering out noise via sparse design. Moreover, we approximate the sparse regression process using a blend of a two-layer MLP and an extensive codebook. This approach not only substantially cuts down on computational costs but also grants SVQ differentiability and training simplicity, resulting in a notable enhancement of performance. Our empirical studies on five spatial-temporal benchmark datasets demonstrate that SVQ achieves state-of-the-art results. Specifically, on the 
    
[^7]: 安全的视觉感知推荐系统：一种对抗图像重构及检测框架

    Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework. (arXiv:2306.07992v1 [cs.CV])

    [http://arxiv.org/abs/2306.07992](http://arxiv.org/abs/2306.07992)

    本文提出了一种对抗图像重构及检测框架来保护视觉感知推荐系统，能够防御局部扰动为特征的对抗攻击并且能够在干净和对抗性图像上进行训练来检测对抗性图像。

    

    随着富含图片等视觉数据与物品关联度增加，视觉感知推荐系统（VARS）已被广泛应用于不同应用领域。最近的研究表明，VARS易受到物品-图像对抗攻击的攻击，这些攻击向与这些物品关联的干净图像添加人类无法感知的扰动。对VARS的攻击为广泛使用VARS的许多应用（如电子商务和社交网络）带来新的安全挑战。如何保护VARS免受此类对抗攻击成为一个关键的问题。目前，尚缺乏系统地研究如何设计针对VARS视觉攻击的安全防御策略。本文提出了一种对抗图像重构及检测框架来保护VARS，我们的方法可以同时(1)通过基于全局视觉传输的图像重构来防御以局部扰动为特征的对抗攻击，(2)使用在少量干净和对抗性图像上训练的检测模型来检测对抗性图像。实验结果表明，我们的框架能够有效地防御各种物品-图像对抗攻击对VARS的影响。

    With rich visual data, such as images, becoming readily associated with items, visually-aware recommendation systems (VARS) have been widely used in different applications. Recent studies have shown that VARS are vulnerable to item-image adversarial attacks, which add human-imperceptible perturbations to the clean images associated with those items. Attacks on VARS pose new security challenges to a wide range of applications such as e-Commerce and social networks where VARS are widely used. How to secure VARS from such adversarial attacks becomes a critical problem. Currently, there is still a lack of systematic study on how to design secure defense strategies against visual attacks on VARS. In this paper, we attempt to fill this gap by proposing an adversarial image reconstruction and detection framework to secure VARS. Our proposed method can simultaneously (1) secure VARS from adversarial attacks characterized by local perturbations by image reconstruction based on global vision tra
    
[^8]: 基于 Foundation Model APIs 的差分隐私合成数据：图片

    Differentially Private Synthetic Data via Foundation Model APIs 1: Images. (arXiv:2305.15560v1 [cs.CV])

    [http://arxiv.org/abs/2305.15560](http://arxiv.org/abs/2305.15560)

    该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。

    

    在当前数据驱动的世界中，生成密切类似于原始私有数据的差分隐私（DP）合成数据是一种可扩展的方法，可减轻隐私问题。与当前为此任务训练定制模型的做法相反，我们旨在通过API生成DP合成数据（DPSDA），其中我们将基础模型视为黑盒并只利用其推理API。这些基于API的、无需训练的方法更容易部署，如最近 API 应用程序的激增所证明的那样。这些方法还可以利用可通过其推理API访问其权重未发布的大型基础模型的能力。但是，由于模型访问更加严格，还需保护API提供商的隐私，这将带来更大的挑战。在本文中，我们提出了一个称为 Private Evolution（PE）的新框架，以解决这个问题，并展示了其在使用基础模型API生成DP合成图像方面的初始实现。PE结合了差分隐私、进化算法和元学习的技术，有效地生成既为DP又与原始图像外观相似的合成图像。我们还在流行的图像数据集如CIFAR-10上评估了我们的框架，并显示我们的方法在效用和隐私方面优于现有的DP图像生成方法。

    Generating differentially private (DP) synthetic data that closely resembles the original private data without leaking sensitive user information is a scalable way to mitigate privacy concerns in the current data-driven world. In contrast to current practices that train customized models for this task, we aim to generate DP Synthetic Data via APIs (DPSDA), where we treat foundation models as blackboxes and only utilize their inference APIs. Such API-based, training-free approaches are easier to deploy as exemplified by the recent surge in the number of API-based apps. These approaches can also leverage the power of large foundation models which are accessible via their inference APIs while the model weights are unreleased. However, this comes with greater challenges due to strictly more restrictive model access and the additional need to protect privacy from the API provider.  In this paper, we present a new framework called Private Evolution (PE) to solve this problem and show its ini
    
[^9]: 自适应离群值优化：用于在线测试时OOD检测的方法

    AUTO: Adaptive Outlier Optimization for Online Test-Time OOD Detection. (arXiv:2303.12267v1 [cs.LG])

    [http://arxiv.org/abs/2303.12267](http://arxiv.org/abs/2303.12267)

    本文提出了一个称为AUTO的方法，在在线测试时利用未标记的在线数据直接提高OOD检测性能。该方法自适应地优化网络参数并在线检测OOD样本，取得了优于现有方法的结果。

    

    在开放世界的应用中，OOD（out-of-distribution）检测是部署机器学习模型的一个关键方面。经验证明，使用辅助离群值训练可以显著提高OOD检测性能。然而，这些离群值通常与测试OOD数据存在分布差距，并且不能覆盖所有可能的测试OOD情况。此外，结合这些离群值还会增加训练的负担。本文提出了一种称为测试时OOD检测的新范式，该范式直接利用未标记的在线数据，以提高OOD检测性能。虽然这种范式很高效，但它也面临着诸如灾难性遗忘等挑战。为了解决这些挑战，我们提出了自适应离群值优化（AUTO），它由内外感知滤波器、ID存储器和语义一致的目标组成。AUTO自适应地从测试数据中挖掘伪ID和伪OOD样本，利用它们来优化网络参数并在线检测OOD样本。实验结果表明，AUTO在各种基准和数据集上始终优于现有方法。

    Out-of-distribution (OOD) detection is a crucial aspect of deploying machine learning models in open-world applications. Empirical evidence suggests that training with auxiliary outliers substantially improves OOD detection. However, such outliers typically exhibit a distribution gap compared to the test OOD data and do not cover all possible test OOD scenarios. Additionally, incorporating these outliers introduces additional training burdens. In this paper, we introduce a novel paradigm called test-time OOD detection, which utilizes unlabeled online data directly at test time to improve OOD detection performance. While this paradigm is efficient, it also presents challenges such as catastrophic forgetting. To address these challenges, we propose adaptive outlier optimization (AUTO), which consists of an in-out-aware filter, an ID memory bank, and a semantically-consistent objective. AUTO adaptively mines pseudo-ID and pseudo-OOD samples from test data, utilizing them to optimize netwo
    
[^10]: DPM-Solver++：用于引导采样的快速扩散概率模型求解器

    DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. (arXiv:2211.01095v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01095](http://arxiv.org/abs/2211.01095)

    DPM-Solver++是一种快速求解器，在扩散概率模型的引导采样中表现出色，可加快样本生成速度。

    

    扩散概率模型 (DPMs) 在高分辨率图像合成等领域表现出色，而 DPM 生成高质量样本所需的关键技术之一是引导采样。现有的快速采样器 DDIM 在引导采样方面表现良好，但需要 100 至 250 步才能生成高质量样本。本文提出了 DPM-Solver++，一种用于加速引导采样的高阶求解器。

    Diffusion probabilistic models (DPMs) have achieved impressive success in high-resolution image synthesis, especially in recent large-scale text-to-image generation applications. An essential technique for improving the sample quality of DPMs is guided sampling, which usually needs a large guidance scale to obtain the best sample quality. The commonly-used fast sampler for guided sampling is DDIM, a first-order diffusion ODE solver that generally needs 100 to 250 steps for high-quality samples. Although recent works propose dedicated high-order solvers and achieve a further speedup for sampling without guidance, their effectiveness for guided sampling has not been well-tested before. In this work, we demonstrate that previous high-order fast samplers suffer from instability issues, and they even become slower than DDIM when the guidance scale grows large. To further speed up guided sampling, we propose DPM-Solver++, a high-order solver for the guided sampling of DPMs. DPM-Solver++ solv
    

