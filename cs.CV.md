# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Separate, Dynamic and Differentiable (SMART) Pruner for Block/Output Channel Pruning on Computer Vision Tasks](https://arxiv.org/abs/2403.19969) | SMART剪枝器引入了独立的、可学习的概率掩码、可微的Top k运算符和动态温度参数技巧，在块和输出通道剪枝任务中显示出优越性。 |
| [^2] | [Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding](https://arxiv.org/abs/2403.17010) | Calib3D是一个从不确定性估计的角度出发，对多个3D场景理解模型进行了全面评估，发现现有模型虽然准确但不可靠，从而阐明了安全关键的背景下的重要性。 |
| [^3] | [If CLIP Could Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions](https://arxiv.org/abs/2403.16442) | 通过新颖的Extract and Explore（EX2）方法，研究发现在视觉-语言模型（VLM）中，重要的特征描述包括非视觉属性，虚假描述影响VLM表示，不同的VLM优先考虑不同的内容。 |
| [^4] | [From Pixels to Insights: A Survey on Automatic Chart Understanding in the Era of Large Foundation Models](https://arxiv.org/abs/2403.12027) | 近年来，随着大型基础模型的兴起，自动图表理解取得了显著进展，本调查论文概述了在这些基础模型背景下图表理解领域的最新发展、挑战和未来方向 |
| [^5] | [Cross-domain and Cross-dimension Learning for Image-to-Graph Transformers](https://arxiv.org/abs/2403.06601) | 该论文提出了一种用于图像到图形转换器的跨域和跨维度迁移学习方法，包括正则化边缘采样损失、领域自适应框架和简单的投影函数，可以解决数据稀缺性问题，并在实验中展示了其实用性。 |
| [^6] | [DeiSAM: Segment Anything with Deictic Prompting](https://arxiv.org/abs/2402.14123) | DeiSAM提出将大型预训练神经网络与可区分逻辑推理器结合，用于指示提示性分割，实现了在复杂场景中对象的分割 |
| [^7] | [Universal Prompt Optimizer for Safe Text-to-Image Generation](https://arxiv.org/abs/2402.10882) | 提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。 |
| [^8] | [CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion](https://arxiv.org/abs/2402.05889) | 该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。 |
| [^9] | [K-space Cold Diffusion: Learning to Reconstruct Accelerated MRI without Noise](https://arxiv.org/abs/2311.10162) | 提出了一种在K空间进行图像退化和恢复的冷扩散模型，无需噪音，能够为加速MRI生成高质量的重建图像 |
| [^10] | [Can We Generate Realistic Hands Only Using Convolution?.](http://arxiv.org/abs/2401.01951) | 本文展示了通过为卷积层提供具有相对$n$维笛卡尔坐标系的单一输入通道，可以缓解图像生成模型无法重现复杂几何特征的问题，显著提高了GAN和VAE生成的手部和面部图像质量。 |
| [^11] | [MRI Field-transfer Reconstruction with Limited Data: Regularization by Neural Style Transfer.](http://arxiv.org/abs/2308.10968) | 本论文通过使用神经风格转换进行正规化，实现了在有限数据条件下从低质量图像重建高质量图像的目标。实验结果验证了该方法在临床MRI扫描中的有效性和潜力。 |
| [^12] | [Differentially Private Synthetic Data via Foundation Model APIs 1: Images.](http://arxiv.org/abs/2305.15560) | 该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。 |
| [^13] | [FPANet: Frequency-based Video Demoireing using Frame-level Post Alignment.](http://arxiv.org/abs/2301.07330) | 该论文提出了一种名为FPANet的新模型，它通过去除各种大小的莫尔纹图案来改善恢复质量，采用多个连续帧提取帧不变内容特征，输出时间一致图像。 |

# 详细

[^1]: 用于计算机视觉任务的分离、动态和可微（SMART）剪枝器

    Separate, Dynamic and Differentiable (SMART) Pruner for Block/Output Channel Pruning on Computer Vision Tasks

    [https://arxiv.org/abs/2403.19969](https://arxiv.org/abs/2403.19969)

    SMART剪枝器引入了独立的、可学习的概率掩码、可微的Top k运算符和动态温度参数技巧，在块和输出通道剪枝任务中显示出优越性。

    

    深度神经网络（DNN）剪枝已被视为减小模型大小、改善推理延迟以及降低DNN加速器功耗的关键策略。在各种剪枝技术中，块和输出通道剪枝在加速硬件性能方面展现出了显著潜力。然而，它们的准确性通常需要进一步改进。为了应对这一挑战，我们介绍了一种名为分离、动态和可微（SMART）剪枝器。该剪枝器利用一个独立的、可学习的概率掩码进行权重重要性排序，采用可微的Top k运算符来实现目标稀疏性，并利用动态温度参数技巧来逃离非稀疏局部极小值。在我们的实验中，SMART剪枝器在块和输出通道剪枝的各种任务和模型上一直表现出优越性。

    arXiv:2403.19969v1 Announce Type: cross  Abstract: Deep Neural Network (DNN) pruning has emerged as a key strategy to reduce model size, improve inference latency, and lower power consumption on DNN accelerators. Among various pruning techniques, block and output channel pruning have shown significant potential in accelerating hardware performance. However, their accuracy often requires further improvement. In response to this challenge, we introduce a separate, dynamic and differentiable (SMART) pruner. This pruner stands out by utilizing a separate, learnable probability mask for weight importance ranking, employing a differentiable Top k operator to achieve target sparsity, and leveraging a dynamic temperature parameter trick to escape from non-sparse local minima. In our experiments, the SMART pruner consistently demonstrated its superiority over existing pruning methods across a wide range of tasks and models on block and output channel pruning. Additionally, we extend our testing
    
[^2]: Calib3D：校准模型偏好以实现可靠的3D场景理解

    Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding

    [https://arxiv.org/abs/2403.17010](https://arxiv.org/abs/2403.17010)

    Calib3D是一个从不确定性估计的角度出发，对多个3D场景理解模型进行了全面评估，发现现有模型虽然准确但不可靠，从而阐明了安全关键的背景下的重要性。

    

    安全关键的3D场景理解任务需要的不仅仅是准确的预测，还需要来自3D感知模型的自信预测。本研究推出了Calib3D，这是一项开创性的工作，旨在从不确定性估计的角度基准和审查3D场景理解模型的可靠性。我们全面评估了28个最先进的模型在10个不同的3D数据集上，揭示了能够处理3D场景理解中的误差不确定性和认知不确定性的有见地的现象。我们发现，尽管现有模型取得了令人印象深刻的准确度水平，但它们经常无法提供可靠的不确定性估计 -- 这个关键的缺陷严重损害了它们在安全敏感环境中的适用性。通过对关键因素（如网络容量、LiDAR表示、光栅分辨率和3D数据增强技术）进行了广泛分析，我们直接将这些方面与模型校准相关联。

    arXiv:2403.17010v1 Announce Type: cross  Abstract: Safety-critical 3D scene understanding tasks necessitate not only accurate but also confident predictions from 3D perception models. This study introduces Calib3D, a pioneering effort to benchmark and scrutinize the reliability of 3D scene understanding models from an uncertainty estimation viewpoint. We comprehensively evaluate 28 state-of-the-art models across 10 diverse 3D datasets, uncovering insightful phenomena that cope with both the aleatoric and epistemic uncertainties in 3D scene understanding. We discover that despite achieving impressive levels of accuracy, existing models frequently fail to provide reliable uncertainty estimates -- a pitfall that critically undermines their applicability in safety-sensitive contexts. Through extensive analysis of key factors such as network capacity, LiDAR representations, rasterization resolutions, and 3D data augmentation techniques, we correlate these aspects directly with the model cal
    
[^3]: 如果CLIP能说话: 通过它们的首选概念描述理解视觉-语言模型的表示

    If CLIP Could Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions

    [https://arxiv.org/abs/2403.16442](https://arxiv.org/abs/2403.16442)

    通过新颖的Extract and Explore（EX2）方法，研究发现在视觉-语言模型（VLM）中，重要的特征描述包括非视觉属性，虚假描述影响VLM表示，不同的VLM优先考虑不同的内容。

    

    最近的研究常常假设视觉-语言模型（VLM）的表示是基于形状等视觉属性。然而，目前尚不清楚VLM在表示概念时在多大程度上将这些信息作为优先考虑对象。我们提出了一种新颖的方法，称为Extract and Explore（EX2），用于刻画VLM的重要文本特征。EX2使用强化学习将一个大型语言模型与VLM首选项对齐，并生成包含VLM重要特征的描述。然后，我们检查这些描述以确定对VLM表示有贡献的特征。我们发现，虽然提供了没有帮助信息的虚假描述（例如，单击放大概念的照片），但在VLM表示中起着重要作用。更重要的是，在信息丰富的描述中，VLM在表示视觉概念时显著依赖非视觉属性（如栖息地）。此外，我们的分析揭示了不同的VLM优先考虑不同的内容。

    arXiv:2403.16442v1 Announce Type: new  Abstract: Recent works often assume that Vision-Language Model (VLM) representations are based on visual attributes like shape. However, it is unclear to what extent VLMs prioritize this information to represent concepts. We propose Extract and Explore (EX2), a novel approach to characterize important textual features for VLMs. EX2 uses reinforcement learning to align a large language model with VLM preferences and generates descriptions that incorporate the important features for the VLM. Then, we inspect the descriptions to identify the features that contribute to VLM representations. We find that spurious descriptions have a major role in VLM representations despite providing no helpful information, e.g., Click to enlarge photo of CONCEPT. More importantly, among informative descriptions, VLMs rely significantly on non-visual attributes like habitat to represent visual concepts. Also, our analysis reveals that different VLMs prioritize differen
    
[^4]: 从像素到洞察: 在大型基础模型时代自动图表理解的调查

    From Pixels to Insights: A Survey on Automatic Chart Understanding in the Era of Large Foundation Models

    [https://arxiv.org/abs/2403.12027](https://arxiv.org/abs/2403.12027)

    近年来，随着大型基础模型的兴起，自动图表理解取得了显著进展，本调查论文概述了在这些基础模型背景下图表理解领域的最新发展、挑战和未来方向

    

    数据可视化以图表形式在数据分析中扮演着关键角色，提供关键洞察并帮助做出明智决策。随着近年大型基础模型的崛起，自动图表理解取得了显著进展。基础模型，如大型语言模型(LLMs)，已经在各种自然语言处理（NLP）任务中实现了革命，并越来越多地应用于图表理解任务。本调查论文全面介绍了最新进展、挑战和未来方向，探讨了这些基础模型背景下图表理解的内容。

    arXiv:2403.12027v1 Announce Type: cross  Abstract: Data visualization in the form of charts plays a pivotal role in data analysis, offering critical insights and aiding in informed decision-making. Automatic chart understanding has witnessed significant advancements with the rise of large foundation models in recent years. Foundation models, such as large language models (LLMs), have revolutionized various natural language processing (NLP) tasks and are increasingly being applied to chart understanding tasks. This survey paper provides a comprehensive overview of the recent developments, challenges, and future directions in chart understanding within the context of these foundation models. The paper begins by defining chart understanding, outlining problem formulations, and discussing fundamental building blocks crucial for studying chart understanding tasks. In the section on tasks and datasets, we explore various tasks within chart understanding and discuss their evaluation metrics a
    
[^5]: 图像到图形变换中的跨域和跨维度学习

    Cross-domain and Cross-dimension Learning for Image-to-Graph Transformers

    [https://arxiv.org/abs/2403.06601](https://arxiv.org/abs/2403.06601)

    该论文提出了一种用于图像到图形转换器的跨域和跨维度迁移学习方法，包括正则化边缘采样损失、领域自适应框架和简单的投影函数，可以解决数据稀缺性问题，并在实验中展示了其实用性。

    

    直接的图像到图形转换是一个具有挑战性的任务，它在单个模型中解决了目标检测和关系预测。由于这个任务的复杂性，在许多领域中很难找到大型训练数据集，这使得训练大型网络具有挑战性。这种数据稀疏性需要建立类似于计算机视觉中最先进技术的预训练策略。在这项工作中，我们引入了一套方法，实现了图像到图形转换器的跨域和跨维度迁移学习。我们提出了(1) 正则化边缘采样损失，用于在不同领域中采样最佳数量的目标关系(边缘)，(2) 一种图像到图形转换器的领域自适应框架，可以对齐不同领域的特征，和(3) 一种简单的投影函数，使我们能够在二维输入数据上预训练三维转换器。我们展示了我们的方法在跨域和跨维度下的实用性。

    arXiv:2403.06601v1 Announce Type: cross  Abstract: Direct image-to-graph transformation is a challenging task that solves object detection and relationship prediction in a single model. Due to the complexity of this task, large training datasets are rare in many domains, which makes the training of large networks challenging. This data sparsity necessitates the establishment of pre-training strategies akin to the state-of-the-art in computer vision. In this work, we introduce a set of methods enabling cross-domain and cross-dimension transfer learning for image-to-graph transformers. We propose (1) a regularized edge sampling loss for sampling the optimal number of object relationships (edges) across domains, (2) a domain adaptation framework for image-to-graph transformers that aligns features from different domains, and (3) a simple projection function that allows us to pretrain 3D transformers on 2D input data. We demonstrate our method's utility in cross-domain and cross-dimension 
    
[^6]: DeiSAM：通过指示提示分割任何内容

    DeiSAM: Segment Anything with Deictic Prompting

    [https://arxiv.org/abs/2402.14123](https://arxiv.org/abs/2402.14123)

    DeiSAM提出将大型预训练神经网络与可区分逻辑推理器结合，用于指示提示性分割，实现了在复杂场景中对象的分割

    

    大规模、预训练的神经网络已经在各种任务中展现出强大的能力，包括零-shot图像分割。为了在复杂场景中识别具体对象，人类本能地依赖于自然语言中的指示性描述，即根据上下文指称某物，比如“在桌子上并在杯子后面的物体”。然而，深度学习方法由于在复杂场景中缺乏推理能力，无法可靠地解释这种指示性表示。为了解决这个问题，我们提出了DeiSAM——将大型预训练神经网络与可区分逻辑推理器相结合，用于指示提示性分割。给定复杂的文本分割描述，DeiSAM利用大型语言模型（LLMs）生成一阶逻辑规则，并对生成的场景图进行可区分的前向推理。随后，DeiSAM通过匹配

    arXiv:2402.14123v1 Announce Type: cross  Abstract: Large-scale, pre-trained neural networks have demonstrated strong capabilities in various tasks, including zero-shot image segmentation. To identify concrete objects in complex scenes, humans instinctively rely on deictic descriptions in natural language, i.e., referring to something depending on the context such as "The object that is on the desk and behind the cup.". However, deep learning approaches cannot reliably interpret such deictic representations due to their lack of reasoning capabilities in complex scenarios. To remedy this issue, we propose DeiSAM -- a combination of large pre-trained neural networks with differentiable logic reasoners -- for deictic promptable segmentation. Given a complex, textual segmentation description, DeiSAM leverages Large Language Models (LLMs) to generate first-order logic rules and performs differentiable forward reasoning on generated scene graphs. Subsequently, DeiSAM segments objects by match
    
[^7]: 通用提示优化器用于安全文本到图像生成

    Universal Prompt Optimizer for Safe Text-to-Image Generation

    [https://arxiv.org/abs/2402.10882](https://arxiv.org/abs/2402.10882)

    提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。

    

    文本到图像（T2I）模型在根据文字提示生成图像方面表现出色。然而，这些模型容易受到不安全输入的影响，从而生成不安全内容，如色情、骚扰和非法活动图像。基于图像检查器、模型微调和嵌入式阻止的现有研究在真实世界应用中不可行。因此，我们提出了第一个用于黑盒场景中安全 T2I 生成的通用提示优化器。

    arXiv:2402.10882v1 Announce Type: cross  Abstract: Text-to-Image (T2I) models have shown great performance in generating images based on textual prompts. However, these models are vulnerable to unsafe input to generate unsafe content like sexual, harassment and illegal-activity images. Existing studies based on image checker, model fine-tuning and embedding blocking are impractical in real-world applications. Hence, \textit{we propose the first universal prompt optimizer for safe T2I generation in black-box scenario}. We first construct a dataset consisting of toxic-clean prompt pairs by GPT-3.5 Turbo. To guide the optimizer to have the ability of converting toxic prompt to clean prompt while preserving semantic information, we design a novel reward function measuring toxicity and text alignment of generated images and train the optimizer through Proximal Policy Optimization. Experiments show that our approach can effectively reduce the likelihood of various T2I models in generating in
    
[^8]: CREMA: 通过有效的模块化适应和融合进行多模态组合视频推理

    CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion

    [https://arxiv.org/abs/2402.05889](https://arxiv.org/abs/2402.05889)

    该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。

    

    尽管在多模态组合推理方法方面取得了令人瞩目的进展，但由于处理固定模态输入并更新许多模型参数，仍然存在灵活性和效率方面的限制。本文解决了这些关键挑战，提出了CREMA，一种用于将任何新的模态注入视频推理的高效且模块化的模态融合框架。我们首先利用现有的预训练模型从给定的视频中增强多种信息模态（如光流、3D点云、音频），而无需额外的人工注释。接下来，我们引入了一个查询转换器，该转换器与每个可以访问的模态相关联，并具有多个参数高效的模块。它将多种模态特征投影到LLM令牌嵌入空间，使模型能够整合不同的数据类型以进行响应生成。此外，我们提出了一个融合模块，用于压缩多模态查询，在LLM中保持计算效率的同时进行融合组合。

    Despite impressive advancements in multimodal compositional reasoning approaches, they are still limited in their flexibility and efficiency by processing fixed modality inputs while updating a lot of model parameters. This paper tackles these critical challenges and proposes CREMA, an efficient and modular modality-fusion framework for injecting any new modality into video reasoning. We first augment multiple informative modalities (such as optical flow, 3D point cloud, audio) from given videos without extra human annotation by leveraging existing pre-trained models. Next, we introduce a query transformer with multiple parameter-efficient modules associated with each accessible modality. It projects diverse modality features to the LLM token embedding space, allowing the model to integrate different data types for response generation. Furthermore, we propose a fusion module designed to compress multimodal queries, maintaining computational efficiency in the LLM while combining additio
    
[^9]: K空间冷扩散：学习在没有噪音的情况下重建加速MRI

    K-space Cold Diffusion: Learning to Reconstruct Accelerated MRI without Noise

    [https://arxiv.org/abs/2311.10162](https://arxiv.org/abs/2311.10162)

    提出了一种在K空间进行图像退化和恢复的冷扩散模型，无需噪音，能够为加速MRI生成高质量的重建图像

    

    基于深度学习的MRI重建模型近年来取得了优异的表现。最近，扩散模型在图像生成、修补、超分辨率、图像编辑等方面表现出色。作为一种通用的扩散模型，冷扩散进一步拓宽了范围，并考虑了围绕任意图像变换构建的模型，例如模糊、下采样等。本文提出了一种在K空间中执行图像退化和恢复的K空间冷扩散模型，无需高斯噪声。我们与多个基于深度学习的MRI重建模型进行比较，并在一个知名的大型开源MRI数据集上进行测试。我们的结果表明，这种新颖的退化方式可以为加速MRI生成高质量的重建图像。

    arXiv:2311.10162v2 Announce Type: replace-cross  Abstract: Deep learning-based MRI reconstruction models have achieved superior performance these days. Most recently, diffusion models have shown remarkable performance in image generation, in-painting, super-resolution, image editing and more. As a generalized diffusion model, cold diffusion further broadens the scope and considers models built around arbitrary image transformations such as blurring, down-sampling, etc. In this paper, we propose a k-space cold diffusion model that performs image degradation and restoration in k-space without the need for Gaussian noise. We provide comparisons with multiple deep learning-based MRI reconstruction models and perform tests on a well-known large open-source MRI dataset. Our results show that this novel way of performing degradation can generate high-quality reconstruction images for accelerated MRI.
    
[^10]: 使用卷积能否仅生成逼真的手部图像？

    Can We Generate Realistic Hands Only Using Convolution?. (arXiv:2401.01951v1 [cs.CV])

    [http://arxiv.org/abs/2401.01951](http://arxiv.org/abs/2401.01951)

    本文展示了通过为卷积层提供具有相对$n$维笛卡尔坐标系的单一输入通道，可以缓解图像生成模型无法重现复杂几何特征的问题，显著提高了GAN和VAE生成的手部和面部图像质量。

    

    长达十年之久，图像生成模型一直无法重现复杂的几何特征，例如人手和手指中所存在的特征，这一问题在图像生成领域一直存在。虽然通过增加模型大小和多样化训练数据集已经取得了一定进展，但这个问题在各种模型中仍然普遍存在，从去噪扩散模型到生成对抗网络（GAN），这指向了底层结构的根本缺陷。在本文中，我们通过为卷积层提供一个单一输入通道，其中包含相对$n$维笛卡尔坐标系，来展示如何缓解这个问题。我们展示了这种方法极大地改善了GAN和变分自动编码器（VAE）生成的手部和面部图像的质量。

    The enduring inability of image generative models to recreate intricate geometric features, such as those present in human hands and fingers has been an ongoing problem in image generation for nearly a decade. While strides have been made by increasing model sizes and diversifying training datasets, this issue remains prevalent across all models, from denoising diffusion models to Generative Adversarial Networks (GAN), pointing to a fundamental shortcoming in the underlying architectures. In this paper, we demonstrate how this problem can be mitigated by augmenting convolution layers geometric capabilities through providing them with a single input channel incorporating the relative $n$-dimensional Cartesian coordinate system. We show that this drastically improves quality of hand and face images generated by GANs and Variational AutoEncoders (VAE).
    
[^11]: 使用有限数据的MRI场转移重建：通过神经风格转换进行正规化

    MRI Field-transfer Reconstruction with Limited Data: Regularization by Neural Style Transfer. (arXiv:2308.10968v1 [cs.CV])

    [http://arxiv.org/abs/2308.10968](http://arxiv.org/abs/2308.10968)

    本论文通过使用神经风格转换进行正规化，实现了在有限数据条件下从低质量图像重建高质量图像的目标。实验结果验证了该方法在临床MRI扫描中的有效性和潜力。

    

    最近的研究表明，使用基于深度学习模型的MRI重建取得了成功。然而，大多数报告的方法都需要在特定任务的大规模数据集上进行训练。通过降噪（RED）正规化是一种将降噪器作为图像重建先验的通用流程。RED的潜力已经在多个与图像相关的任务（如降噪、去模糊和超分辨率）中得到了证明。本文提出了一种通过神经风格转换（RNST）方法进行正规化的方法，进一步利用神经转移和降噪引擎的先验信息。这使得RNST能够从有噪声的低质量图像中重建出高质量图像，图像风格和有限数据不同。我们使用1.5T和3T的临床MRI扫描验证了RNST，并且显示RNST可以显著提高图像质量。我们的结果突显了RNST框架在MRI重建和有限数据重建任务中的能力。

    Recent works have demonstrated success in MRI reconstruction using deep learning-based models. However, most reported approaches require training on a task-specific, large-scale dataset. Regularization by denoising (RED) is a general pipeline which embeds a denoiser as a prior for image reconstruction. The potential of RED has been demonstrated for multiple image-related tasks such as denoising, deblurring and super-resolution. In this work, we propose a regularization by neural style transfer (RNST) method to further leverage the priors from the neural transfer and denoising engine. This enables RNST to reconstruct a high-quality image from a noisy low-quality image with different image styles and limited data. We validate RNST with clinical MRI scans from 1.5T and 3T and show that RNST can significantly boost image quality. Our results highlight the capability of the RNST framework for MRI reconstruction and the potential for reconstruction tasks with limited data.
    
[^12]: 基于 Foundation Model APIs 的差分隐私合成数据：图片

    Differentially Private Synthetic Data via Foundation Model APIs 1: Images. (arXiv:2305.15560v1 [cs.CV])

    [http://arxiv.org/abs/2305.15560](http://arxiv.org/abs/2305.15560)

    该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。

    

    在当前数据驱动的世界中，生成密切类似于原始私有数据的差分隐私（DP）合成数据是一种可扩展的方法，可减轻隐私问题。与当前为此任务训练定制模型的做法相反，我们旨在通过API生成DP合成数据（DPSDA），其中我们将基础模型视为黑盒并只利用其推理API。这些基于API的、无需训练的方法更容易部署，如最近 API 应用程序的激增所证明的那样。这些方法还可以利用可通过其推理API访问其权重未发布的大型基础模型的能力。但是，由于模型访问更加严格，还需保护API提供商的隐私，这将带来更大的挑战。在本文中，我们提出了一个称为 Private Evolution（PE）的新框架，以解决这个问题，并展示了其在使用基础模型API生成DP合成图像方面的初始实现。PE结合了差分隐私、进化算法和元学习的技术，有效地生成既为DP又与原始图像外观相似的合成图像。我们还在流行的图像数据集如CIFAR-10上评估了我们的框架，并显示我们的方法在效用和隐私方面优于现有的DP图像生成方法。

    Generating differentially private (DP) synthetic data that closely resembles the original private data without leaking sensitive user information is a scalable way to mitigate privacy concerns in the current data-driven world. In contrast to current practices that train customized models for this task, we aim to generate DP Synthetic Data via APIs (DPSDA), where we treat foundation models as blackboxes and only utilize their inference APIs. Such API-based, training-free approaches are easier to deploy as exemplified by the recent surge in the number of API-based apps. These approaches can also leverage the power of large foundation models which are accessible via their inference APIs while the model weights are unreleased. However, this comes with greater challenges due to strictly more restrictive model access and the additional need to protect privacy from the API provider.  In this paper, we present a new framework called Private Evolution (PE) to solve this problem and show its ini
    
[^13]: FPANet: 基于频率的视频去莫尔纹技术，使用帧级后对齐

    FPANet: Frequency-based Video Demoireing using Frame-level Post Alignment. (arXiv:2301.07330v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.07330](http://arxiv.org/abs/2301.07330)

    该论文提出了一种名为FPANet的新模型，它通过去除各种大小的莫尔纹图案来改善恢复质量，采用多个连续帧提取帧不变内容特征，输出时间一致图像。

    

    重叠网格模式之间的干扰会导致莫尔纹，从而降低普通数码相机捕捉数字显示屏的图像的视觉质量。该论文提出了一种名为FPANet的新模型，它学习频率和空间域中的滤波器，通过去除各种大小的莫尔纹图案来改善恢复质量。此外，模型使用多个连续帧，学习提取帧不变内容特征，并输出更好质量的时间一致图像。

    Interference between overlapping gird patterns creates moire patterns, degrading the visual quality of an image that captures a screen of a digital display device by an ordinary digital camera. Removing such moire patterns is challenging due to their complex patterns of diverse sizes and color distortions. Existing approaches mainly focus on filtering out in the spatial domain, failing to remove a large-scale moire pattern. In this paper, we propose a novel model called FPANet that learns filters in both frequency and spatial domains, improving the restoration quality by removing various sizes of moire patterns. To further enhance, our model takes multiple consecutive frames, learning to extract frame-invariant content features and outputting better quality temporally consistent images. We demonstrate the effectiveness of our proposed method with a publicly available large-scale dataset, observing that ours outperforms the state-of-the-art approaches, including ESDNet, VDmoire, MBCNN, 
    

