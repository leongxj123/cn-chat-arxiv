# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Counterfactual Image Generation](https://arxiv.org/abs/2403.20287) | 提出了一个针对对照图像生成方法的基准测试框架，包含评估对照多个方面的度量标准以及评估三种不同类型的条件图像生成模型性能。 |
| [^2] | [Improving Forward Compatibility in Class Incremental Learning by Increasing Representation Rank and Feature Richness](https://arxiv.org/abs/2403.15517) | 该研究引入了一种基于有效秩的特征丰富性增强（RFR）方法，通过增加表示的有效秩，实现了提高前向兼容性的目标。同时，在后向兼容性和前向兼容性方面取得了双重目标。 |
| [^3] | [Advanced Artificial Intelligence Algorithms in Cochlear Implants: Review of Healthcare Strategies, Challenges, and Perspectives](https://arxiv.org/abs/2403.15442) | 人工智能在提高植入式听觉设备的语音质量方面具有前瞻性，并通过先进的信号处理技术以及应对多源语音和环境噪音挑战等方法来克服语音失真问题 |
| [^4] | [Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models](https://arxiv.org/abs/2403.09792) | 论文研究了多模态大语言模型对齐问题，揭示了图像输入对模型的漏洞，并提出了一种利用图像隐藏恶意意图、成功破解现有模型的方法。 |
| [^5] | [Content-aware Masked Image Modeling Transformer for Stereo Image Compression](https://arxiv.org/abs/2403.08505) | 提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。 |
| [^6] | [Improve Robustness of Eye Disease Detection by including Learnable Probabilistic Discrete Latent Variables into Machine Learning Models](https://arxiv.org/abs/2402.16865) | 通过引入可学习的概率离散潜变量，该研究提出了一种新颖的眼部疾病检测方法，利用生成流网络来学习眼底图像中眼部疾病的后验分布，提高了鲁棒性和泛化能力。 |
| [^7] | [Explainable Classification Techniques for Quantum Dot Device Measurements](https://arxiv.org/abs/2402.13699) | 提出了一种基于合成数据的的可解释特征技术，利用可解释性提升机（EBMs）实现了在量子点调谐中较高的可解释性和准确性。 |
| [^8] | [SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2401.11791) | SemPLeS框架利用语义提示学习解决弱监督语义分割中的问题，通过学习有效提示来增强分割区域与目标对象类别之间的语义对齐。 |
| [^9] | [Quilt-1M: One Million Image-Text Pairs for Histopathology.](http://arxiv.org/abs/2306.11207) | 本文介绍了一个名为 Quilt-1M 的癌症组织学图像和文字对的百万数据集，并利用 YouTube 上的专家医生教程视频为主要来源。这个数据集将使得癌症组织学领域的表征学习取得类似于其他领域的进展。 |
| [^10] | [Privacy-Preserving CNN Training with Transfer Learning.](http://arxiv.org/abs/2304.03807) | 本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。 |

# 详细

[^1]: 基准对照图像生成

    Benchmarking Counterfactual Image Generation

    [https://arxiv.org/abs/2403.20287](https://arxiv.org/abs/2403.20287)

    提出了一个针对对照图像生成方法的基准测试框架，包含评估对照多个方面的度量标准以及评估三种不同类型的条件图像生成模型性能。

    

    对照图像生成在理解变量因果关系方面具有关键作用，在解释性和生成无偏合成数据方面有应用。然而，评估图像生成本身就是一个长期存在的挑战。对于评估对照生成的需求进一步加剧了这一挑战，因为根据定义，对照情景是没有可观测基准事实的假设情况。本文提出了一个旨在对照图像生成方法进行基准测试的新颖综合框架。我们结合了侧重于评估对照的不同方面的度量标准，例如组成、有效性、干预的最小性和图像逼真度。我们评估了基于结构因果模型范式的三种不同条件图像生成模型类型的性能。我们的工作还配备了一个用户友好的Python软件包，可以进一步评估。

    arXiv:2403.20287v1 Announce Type: cross  Abstract: Counterfactual image generation is pivotal for understanding the causal relations of variables, with applications in interpretability and generation of unbiased synthetic data. However, evaluating image generation is a long-standing challenge in itself. The need to evaluate counterfactual generation compounds on this challenge, precisely because counterfactuals, by definition, are hypothetical scenarios without observable ground truths. In this paper, we present a novel comprehensive framework aimed at benchmarking counterfactual image generation methods. We incorporate metrics that focus on evaluating diverse aspects of counterfactuals, such as composition, effectiveness, minimality of interventions, and image realism. We assess the performance of three distinct conditional image generation model types, based on the Structural Causal Model paradigm. Our work is accompanied by a user-friendly Python package which allows to further eval
    
[^2]: 通过增加表示秩和特征丰富性来改善类增量学习中的前向兼容性

    Improving Forward Compatibility in Class Incremental Learning by Increasing Representation Rank and Feature Richness

    [https://arxiv.org/abs/2403.15517](https://arxiv.org/abs/2403.15517)

    该研究引入了一种基于有效秩的特征丰富性增强（RFR）方法，通过增加表示的有效秩，实现了提高前向兼容性的目标。同时，在后向兼容性和前向兼容性方面取得了双重目标。

    

    类增量学习（CIL）是连续学习中的一个重要子领域，旨在使模型能够在保留先前任务知识的同时逐渐学习新的分类任务。本研究引入了一种基于有效秩的特征丰富性增强（RFR）方法，旨在提高前向兼容性。具体而言，该方法通过增加基础阶段表示的有效秩，从而有助于合并更多与未见新任务相关的信息特征。因此，RFR在后向兼容性和前向兼容性方面实现了双重目标：最大限度地减小特征提取器的影响。

    arXiv:2403.15517v1 Announce Type: new  Abstract: Class Incremental Learning (CIL) constitutes a pivotal subfield within continual learning, aimed at enabling models to progressively learn new classification tasks while retaining knowledge obtained from prior tasks. Although previous studies have predominantly focused on backward compatible approaches to mitigate catastrophic forgetting, recent investigations have introduced forward compatible methods to enhance performance on novel tasks and complement existing backward compatible methods. In this study, we introduce an effective-Rank based Feature Richness enhancement (RFR) method, designed for improving forward compatibility. Specifically, this method increases the effective rank of representations during the base session, thereby facilitating the incorporation of more informative features pertinent to unseen novel tasks. Consequently, RFR achieves dual objectives in backward and forward compatibility: minimizing feature extractor mo
    
[^3]: 人工智能在耳蜗植入装置中的先进算法：医疗策略、挑战和展望综述

    Advanced Artificial Intelligence Algorithms in Cochlear Implants: Review of Healthcare Strategies, Challenges, and Perspectives

    [https://arxiv.org/abs/2403.15442](https://arxiv.org/abs/2403.15442)

    人工智能在提高植入式听觉设备的语音质量方面具有前瞻性，并通过先进的信号处理技术以及应对多源语音和环境噪音挑战等方法来克服语音失真问题

    

    arXiv:2403.15442v1 公告类型: 跨领域 摘要: 自动语音识别（ASR）在我们的日常生活中发挥着至关重要的作用，不仅为与机器交互提供了便利，还为部分或完全听力受损的个体提供了沟通的机会。这一过程涉及以模拟形式接收语音信号，然后通过各种信号处理算法使其与容量有限的设备（如CI）兼容。然而，这些配备有有限数量电极的植入装置在合成过程中往往导致语音失真。尽管研究人员在使用各种最先进的信号处理技术改善接收到的语音质量方面做出了努力，但在涉及多个语音源、环境噪声和其他情况的场景中，挑战仍然存在。新人工智能（AI）方法的出现引入了先进的策略来解决这些限制。

    arXiv:2403.15442v1 Announce Type: cross  Abstract: Automatic speech recognition (ASR) plays a pivotal role in our daily lives, offering utility not only for interacting with machines but also for facilitating communication for individuals with either partial or profound hearing impairments. The process involves receiving the speech signal in analogue form, followed by various signal processing algorithms to make it compatible with devices of limited capacity, such as cochlear implants (CIs). Unfortunately, these implants, equipped with a finite number of electrodes, often result in speech distortion during synthesis. Despite efforts by researchers to enhance received speech quality using various state-of-the-art signal processing techniques, challenges persist, especially in scenarios involving multiple sources of speech, environmental noise, and other circumstances. The advent of new artificial intelligence (AI) methods has ushered in cutting-edge strategies to address the limitations
    
[^4]: 图像是对齐的软肋：利用视觉漏洞破解多模态大语言模型

    Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models

    [https://arxiv.org/abs/2403.09792](https://arxiv.org/abs/2403.09792)

    论文研究了多模态大语言模型对齐问题，揭示了图像输入对模型的漏洞，并提出了一种利用图像隐藏恶意意图、成功破解现有模型的方法。

    

    在这篇论文中，我们研究了多模态大语言模型（MLLMs）的无害对齐问题。我们对代表性的MLLMs的无害性能进行了系统的实证分析，并揭示了图像输入对MLLMs造成的对齐漏洞。受此启发，我们提出了一种名为HADES的新型破解方法，利用精心制作的图像隐藏和放大文本输入中的恶意意图的有害性。实验结果表明，HADES可以有效地破解现有的MLLMs，为LLaVA-1.5实现了90.26％的平均攻击成功率（ASR），为Gemini Pro Vision实现了71.60％的ASR。我们的代码和数据将公开发布。

    arXiv:2403.09792v1 Announce Type: cross  Abstract: In this paper, we study the harmlessness alignment problem of multimodal large language models~(MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate~(ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.
    
[^5]: 面向内容感知的掩码图像建模变压器用于立体图像压缩

    Content-aware Masked Image Modeling Transformer for Stereo Image Compression

    [https://arxiv.org/abs/2403.08505](https://arxiv.org/abs/2403.08505)

    提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。

    

    现有基于学习的立体图像编解码器采用了复杂的转换方法，但在编码潜在表示时却采用了从单个图像编解码器导出的简单熵模型。然而，这些熵模型难以有效捕捉立体图像固有的空间-视差特征，导致亚最优的率失真结果。本文提出了一种名为CAMSIC的立体图像压缩框架。 CAMSIC 独立地将每个图像转换为潜在表示，并采用强大的无解码器变压器熵模型来捕捉空间和视差依赖关系，引入了一种新颖的面向内容感知的掩码图像建模（MIM）技术。我们的面向内容感知的MIM促进了先验信息与估计令牌之间的高效双向交互，自然地消除了额外的Transformer解码器的需求。实验证明，我们的立体图像编解码器实现了最先进的率失真结果。

    arXiv:2403.08505v1 Announce Type: cross  Abstract: Existing learning-based stereo image codec adopt sophisticated transformation with simple entropy models derived from single image codecs to encode latent representations. However, those entropy models struggle to effectively capture the spatial-disparity characteristics inherent in stereo images, which leads to suboptimal rate-distortion results. In this paper, we propose a stereo image compression framework, named CAMSIC. CAMSIC independently transforms each image to latent representation and employs a powerful decoder-free Transformer entropy model to capture both spatial and disparity dependencies, by introducing a novel content-aware masked image modeling (MIM) technique. Our content-aware MIM facilitates efficient bidirectional interaction between prior information and estimated tokens, which naturally obviates the need for an extra Transformer decoder. Experiments show that our stereo image codec achieves state-of-the-art rate-d
    
[^6]: 通过将可学习的概率离散潜变量引入机器学习模型来提高眼部疾病检测的鲁棒性

    Improve Robustness of Eye Disease Detection by including Learnable Probabilistic Discrete Latent Variables into Machine Learning Models

    [https://arxiv.org/abs/2402.16865](https://arxiv.org/abs/2402.16865)

    通过引入可学习的概率离散潜变量，该研究提出了一种新颖的眼部疾病检测方法，利用生成流网络来学习眼底图像中眼部疾病的后验分布，提高了鲁棒性和泛化能力。

    

    眼部疾病从糖尿病性视网膜病变到青光眼等，由于其高发病率和可能导致视力损害，构成了一个重要的公共卫生挑战。及早和准确的诊断对于有效治疗和管理至关重要。近年来，深度学习模型已经成为分析医学图像（包括眼部图像）的强大工具。然而，模型的解释性和不确定性估计方面仍然存在挑战，这对临床决策至关重要。本研究引入了GFlowOut的新颖应用，利用生成流网络（GFlowNets）的概率框架来学习关于辍学掩码的后验分布，用于使用眼底图像对眼部疾病进行分类和分析。我们开发了一种稳健且具有普适性的方法，利用以ResNet18和ViT模型为主干的GFlowOut来识别各种眼部状况。

    arXiv:2402.16865v1 Announce Type: cross  Abstract: Ocular diseases, ranging from diabetic retinopathy to glaucoma, present a significant public health challenge due to their prevalence and potential for causing vision impairment. Early and accurate diagnosis is crucial for effective treatment and management.In recent years, deep learning models have emerged as powerful tools for analysing medical images, including ocular imaging . However, challenges persist in model interpretability and uncertainty estimation, which are critical for clinical decision-making. This study introduces a novel application of GFlowOut, leveraging the probabilistic framework of Generative Flow Networks (GFlowNets) to learn the posterior distribution over dropout masks, for the classification and analysis of ocular diseases using eye fundus images. We develop a robust and generalizable method that utilizes GFlowOut integrated with ResNet18 and ViT models as backbone in identifying various ocular conditions. Th
    
[^7]: 可解释的量子点器件测量分类技术

    Explainable Classification Techniques for Quantum Dot Device Measurements

    [https://arxiv.org/abs/2402.13699](https://arxiv.org/abs/2402.13699)

    提出了一种基于合成数据的的可解释特征技术，利用可解释性提升机（EBMs）实现了在量子点调谐中较高的可解释性和准确性。

    

    在物理科学中，对图像数据的稳健特征表示需求增加：图像采集，在广义上指二维数据，现在在许多领域广泛应用，包括我们在此考虑的量子信息科学。虽然在这些情况下广泛使用传统图像特征，但它们的使用正在迅速被神经网络技术所取代，后者往往以牺牲可解释性为代价换取高准确性。为了弥合这种权衡，我们提出了一种基于合成数据的技术，可以产生可解释的特征。我们利用可解释性提升机（EBMs）展示，这种方法提供了卓越的可解释性，并且不会降低准确性。具体而言，我们展示了在量子点调谐的背景下，这种技术带来了实质性的益处，当前发展阶段需要人类干预。

    arXiv:2402.13699v1 Announce Type: cross  Abstract: In the physical sciences, there is an increased need for robust feature representations of image data: image acquisition, in the generalized sense of two-dimensional data, is now widespread across a large number of fields, including quantum information science, which we consider here. While traditional image features are widely utilized in such cases, their use is rapidly being supplanted by Neural Network-based techniques that often sacrifice explainability in exchange for high accuracy. To ameliorate this trade-off, we propose a synthetic data-based technique that results in explainable features. We show, using Explainable Boosting Machines (EBMs), that this method offers superior explainability without sacrificing accuracy. Specifically, we show that there is a meaningful benefit to this technique in the context of quantum dot tuning, where human intervention is necessary at the current stage of development.
    
[^8]: SemPLeS: 语义提示学习用于弱监督语义分割

    SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation

    [https://arxiv.org/abs/2401.11791](https://arxiv.org/abs/2401.11791)

    SemPLeS框架利用语义提示学习解决弱监督语义分割中的问题，通过学习有效提示来增强分割区域与目标对象类别之间的语义对齐。

    

    弱监督语义分割（WSSS）旨在利用仅具有图像级监督的图像数据来训练分割模型。由于无法获得精确的像素级标注，现有方法通常侧重于通过优化CAM样式的热图来生成用于训练分割模型的伪标记。然而，生成的热图可能仅捕获对象类别的具有区分性的图像区域或相关的共同出现的背景。为解决这些问题，我们提出了一种用于WSSS的语义提示学习（SemPLeS）框架，该框架学习有效地提示CLIP潜空间以增强分割区域与目标对象类别之间的语义对准。具体而言，我们提出了对比提示学习和提示引导的语义细化，以学习适当描述和抑制与每个目标对象类别相关的共同出现的背景的提示。

    arXiv:2401.11791v2 Announce Type: replace-cross  Abstract: Weakly-Supervised Semantic Segmentation (WSSS) aims to train segmentation models using image data with only image-level supervision. Since precise pixel-level annotations are not accessible, existing methods typically focus on producing pseudo masks for training segmentation models by refining CAM-like heatmaps. However, the produced heatmaps may capture only the discriminative image regions of object categories or the associated co-occurring backgrounds. To address the issues, we propose a Semantic Prompt Learning for WSSS (SemPLeS) framework, which learns to effectively prompt the CLIP latent space to enhance the semantic alignment between the segmented regions and the target object categories. More specifically, we propose Contrastive Prompt Learning and Prompt-guided Semantic Refinement to learn the prompts that adequately describe and suppress the co-occurring backgrounds associated with each target object category. In thi
    
[^9]: Quilt-1M: 癌症组织学图像文字对的百万数据集

    Quilt-1M: One Million Image-Text Pairs for Histopathology. (arXiv:2306.11207v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.11207](http://arxiv.org/abs/2306.11207)

    本文介绍了一个名为 Quilt-1M 的癌症组织学图像和文字对的百万数据集，并利用 YouTube 上的专家医生教程视频为主要来源。这个数据集将使得癌症组织学领域的表征学习取得类似于其他领域的进展。

    

    多模态应用的加速使得在线图像和文字数据大量涌现，但医学领域（特别是癌症组织学）类似的数据却很稀少，这阻碍了医学领域的进展。本文利用YouTube上的专家医生教程视频，从中选择了 1,087 小时的医学组织学视频，以此自动筛选出共包含 768,826 个癌症组织学图像及其对应的文字对的 Quilt 数据集。

    Recent accelerations in multi-modal applications have been made possible with the plethora of image and text data available online. However, the scarcity of analogous data in the medical field, specifically in histopathology, has halted comparable progress. To enable similar representation learning for histopathology, we turn to YouTube, an untapped resource of videos, offering $1,087$ hours of valuable educational histopathology videos from expert clinicians. From YouTube, we curate Quilt: a large-scale vision-language dataset consisting of $768,826$ image and text pairs. Quilt was automatically curated using a mixture of models, including large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition. In comparison, the most comprehensive datasets curated for histopathology amass only around $200$K samples. We combine Quilt with datasets from other sources, including Twitter, research papers, and the internet in general, to create an even l
    
[^10]: 使用迁移学习实现隐私保护的CNN训练

    Privacy-Preserving CNN Training with Transfer Learning. (arXiv:2304.03807v1 [cs.CR])

    [http://arxiv.org/abs/2304.03807](http://arxiv.org/abs/2304.03807)

    本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。

    

    隐私保护的神经网络推理已经得到很好的研究，同时保持同态CNN训练仍然是一项挑战性的任务。在本文中，我们提出了一种实用的解决方案来实现基于同态加密技术的隐私保护CNN训练。据我们所知，这是第一次成功突破这个难题，以前没有任何工作达到这个目标。采用了几种技术：（1）通过迁移学习，可以将隐私保护的CNN训练简化为同态神经网络训练，甚至是多类逻辑回归（MLR）训练；（2）通过更快的梯度变体$\texttt{Quadratic Gradient}$，应用于MLR的增强梯度方法，在收敛速度方面具有最先进的性能；（3）我们采用数学中的变换思想，将加密域中的近似Softmax函数转换成已经研究过的逼近方法，从而得到更好的结果。

    Privacy-preserving nerual network inference has been well studied while homomorphic CNN training still remains an open challenging task. In this paper, we present a practical solution to implement privacy-preserving CNN training based on mere Homomorphic Encryption (HE) technique. To our best knowledge, this is the first attempt successfully to crack this nut and no work ever before has achieved this goal. Several techniques combine to make it done: (1) with transfer learning, privacy-preserving CNN training can be reduced to homomorphic neural network training, or even multiclass logistic regression (MLR) training; (2) via a faster gradient variant called $\texttt{Quadratic Gradient}$, an enhanced gradient method for MLR with a state-of-the-art performance in converge speed is applied in this work to achieve high performance; (3) we employ the thought of transformation in mathematics to transform approximating Softmax function in encryption domain to the well-studied approximation of 
    

