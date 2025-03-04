# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sine Activated Low-Rank Matrices for Parameter Efficient Learning](https://arxiv.org/abs/2403.19243) | 整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。 |
| [^2] | [Bidirectional Consistency Models](https://arxiv.org/abs/2403.18035) | 提出了双向一致性模型（BCM），学习一个神经网络，能够实现沿着概率流常微分方程前向和后向遍历，从而有效地统一了生成和编辑图像等任务。 |
| [^3] | [Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding](https://arxiv.org/abs/2403.17010) | Calib3D是一个从不确定性估计的角度出发，对多个3D场景理解模型进行了全面评估，发现现有模型虽然准确但不可靠，从而阐明了安全关键的背景下的重要性。 |
| [^4] | [A Decade's Battle on Dataset Bias: Are We There Yet?](https://arxiv.org/abs/2403.08632) | 现代神经网络在分类来自不同数据集的图像方面表现出色，具有可推广和可转移的语义特征，挑战了传统的数据集偏见认知。 |
| [^5] | [CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations](https://arxiv.org/abs/2402.04236) | 本文介绍了CogCoM，一个具备操作链机制的大规模视觉语言模型，通过一系列操作解决视觉问题，并以其证据性的视觉推理能力实现忠实的响应。 |
| [^6] | [SoK: Facial Deepfake Detectors.](http://arxiv.org/abs/2401.04364) | 本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。 |
| [^7] | [Assessing Robustness via Score-Based Adversarial Image Generation.](http://arxiv.org/abs/2310.04285) | 本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。 |
| [^8] | [Representation Engineering: A Top-Down Approach to AI Transparency.](http://arxiv.org/abs/2310.01405) | 这项研究介绍了一种名为表示工程化（RepE）的自上而下方法，通过借鉴认知神经科学的见解，提供了一种增强AI系统透明性的解决方案。该方法将集群级别的表示放在分析的核心，为监测和操纵深度神经网络中的高级认知现象提供了新的方法，并展示了在解决与安全相关的问题上的潜力。 |
| [^9] | [End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection.](http://arxiv.org/abs/2306.12033) | 这项研究提出了一种名为ST-SSAD的新方法，可以系统地调整数据增强的超参数，从而有助于提高自我监督异常检测（SSAD）的性能。 |
| [^10] | [Open-radiomics: A Collection of Standardized Datasets and a Technical Protocol for Reproducible Radiomics Machine Learning Pipelines.](http://arxiv.org/abs/2207.14776) | 本研究提出了一套开放放射组学数据集和技术协议，旨在解决放射组学在结果可重复性和可访问性方面所面临的挑战。通过在BraTS 2020数据集上进行实验，研究了放射组学特征提取对结果可重复性的影响。 |

# 详细

[^1]: 用正弦激活的低秩矩阵实现参数高效学习

    Sine Activated Low-Rank Matrices for Parameter Efficient Learning

    [https://arxiv.org/abs/2403.19243](https://arxiv.org/abs/2403.19243)

    整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。

    

    低秩分解已经成为在神经网络架构中增强参数效率的重要工具，在机器学习的各种应用中越来越受到关注。这些技术显著降低了参数数量，取得了简洁性和性能之间的平衡。然而，一个常见的挑战是在参数效率和模型准确性之间做出妥协，参数减少往往导致准确性不及完整秩对应模型。在这项工作中，我们提出了一个创新的理论框架，在低秩分解过程中整合了一个正弦函数。这种方法不仅保留了低秩方法的参数效率特性的好处，还增加了分解的秩，从而提高了模型的准确性。我们的方法被证明是现有低秩模型的一种适应性增强，正如其成功证实的那样。

    arXiv:2403.19243v1 Announce Type: new  Abstract: Low-rank decomposition has emerged as a vital tool for enhancing parameter efficiency in neural network architectures, gaining traction across diverse applications in machine learning. These techniques significantly lower the number of parameters, striking a balance between compactness and performance. However, a common challenge has been the compromise between parameter efficiency and the accuracy of the model, where reduced parameters often lead to diminished accuracy compared to their full-rank counterparts. In this work, we propose a novel theoretical framework that integrates a sinusoidal function within the low-rank decomposition process. This approach not only preserves the benefits of the parameter efficiency characteristic of low-rank methods but also increases the decomposition's rank, thereby enhancing model accuracy. Our method proves to be an adaptable enhancement for existing low-rank models, as evidenced by its successful 
    
[^2]: 双向一致性模型

    Bidirectional Consistency Models

    [https://arxiv.org/abs/2403.18035](https://arxiv.org/abs/2403.18035)

    提出了双向一致性模型（BCM），学习一个神经网络，能够实现沿着概率流常微分方程前向和后向遍历，从而有效地统一了生成和编辑图像等任务。

    

    扩散模型（DMs）通过迭代去噪一个随机向量能够生成非常高质量的样本，这个过程对应于沿着概率流常微分方程（PF ODE）移动。有趣的是，DMs还可以通过沿着PF ODE向后移动将输入图像转换为噪声，这是下游任务（如插值和图像编辑）的关键操作。然而，这一过程的迭代性质限制了其速度，阻碍了其更广泛的应用。最近，一致性模型（CMs）已经出现，以解决这一挑战，通过近似PF ODE的积分，从而避免了需要迭代。然而，缺乏显式ODE求解器使得反演过程复杂化。为了解决这个问题，我们引入了双向一致性模型（BCM），学习单个神经网络，能够同时实现沿着PF ODE的前向和后向遍历，有效地统一生成和

    arXiv:2403.18035v1 Announce Type: new  Abstract: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, thereby bypassing the need to iterate. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation an
    
[^3]: Calib3D：校准模型偏好以实现可靠的3D场景理解

    Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding

    [https://arxiv.org/abs/2403.17010](https://arxiv.org/abs/2403.17010)

    Calib3D是一个从不确定性估计的角度出发，对多个3D场景理解模型进行了全面评估，发现现有模型虽然准确但不可靠，从而阐明了安全关键的背景下的重要性。

    

    安全关键的3D场景理解任务需要的不仅仅是准确的预测，还需要来自3D感知模型的自信预测。本研究推出了Calib3D，这是一项开创性的工作，旨在从不确定性估计的角度基准和审查3D场景理解模型的可靠性。我们全面评估了28个最先进的模型在10个不同的3D数据集上，揭示了能够处理3D场景理解中的误差不确定性和认知不确定性的有见地的现象。我们发现，尽管现有模型取得了令人印象深刻的准确度水平，但它们经常无法提供可靠的不确定性估计 -- 这个关键的缺陷严重损害了它们在安全敏感环境中的适用性。通过对关键因素（如网络容量、LiDAR表示、光栅分辨率和3D数据增强技术）进行了广泛分析，我们直接将这些方面与模型校准相关联。

    arXiv:2403.17010v1 Announce Type: cross  Abstract: Safety-critical 3D scene understanding tasks necessitate not only accurate but also confident predictions from 3D perception models. This study introduces Calib3D, a pioneering effort to benchmark and scrutinize the reliability of 3D scene understanding models from an uncertainty estimation viewpoint. We comprehensively evaluate 28 state-of-the-art models across 10 diverse 3D datasets, uncovering insightful phenomena that cope with both the aleatoric and epistemic uncertainties in 3D scene understanding. We discover that despite achieving impressive levels of accuracy, existing models frequently fail to provide reliable uncertainty estimates -- a pitfall that critically undermines their applicability in safety-sensitive contexts. Through extensive analysis of key factors such as network capacity, LiDAR representations, rasterization resolutions, and 3D data augmentation techniques, we correlate these aspects directly with the model cal
    
[^4]: 十年数据集偏见之战：我们已经成功了吗？

    A Decade's Battle on Dataset Bias: Are We There Yet?

    [https://arxiv.org/abs/2403.08632](https://arxiv.org/abs/2403.08632)

    现代神经网络在分类来自不同数据集的图像方面表现出色，具有可推广和可转移的语义特征，挑战了传统的数据集偏见认知。

    

    我们在新时代重新审视Torralba和Efros十年前提出的“数据集分类”实验，在拥有大规模、多样化和希望更少偏见的数据集以及更强大的神经网络架构的新时代。令人惊讶的是，我们观察到现代神经网络能够在分类图像来自哪个数据集方面取得出色的准确性：例如，对于包含YFCC、CC和DataComp数据集的三分类问题的验证数据，我们报告84.7%的准确性。我们进一步的实验表明，这样的数据集分类器可以学习到可推广和可转移的语义特征，这不能简单地解释为记忆。我们希望我们的发现能激励社区重新思考涉及数据集偏见和模型能力的问题。

    arXiv:2403.08632v1 Announce Type: cross  Abstract: We revisit the "dataset classification" experiment suggested by Torralba and Efros a decade ago, in the new era with large-scale, diverse, and hopefully less biased datasets as well as more capable neural network architectures. Surprisingly, we observe that modern neural networks can achieve excellent accuracy in classifying which dataset an image is from: e.g., we report 84.7% accuracy on held-out validation data for the three-way classification problem consisting of the YFCC, CC, and DataComp datasets. Our further experiments show that such a dataset classifier could learn semantic features that are generalizable and transferable, which cannot be simply explained by memorization. We hope our discovery will inspire the community to rethink the issue involving dataset bias and model capabilities.
    
[^5]: CogCoM: 通过一系列的操作训练大规模视觉语言模型，并深入细节

    CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations

    [https://arxiv.org/abs/2402.04236](https://arxiv.org/abs/2402.04236)

    本文介绍了CogCoM，一个具备操作链机制的大规模视觉语言模型，通过一系列操作解决视觉问题，并以其证据性的视觉推理能力实现忠实的响应。

    

    视觉语言模型（VLM）通过广泛的训练，在将视觉指令与答案对齐方面展示了广泛的可行性。然而，这种确定性的对齐导致模型忽视了关键的视觉推理，并导致在细致的视觉问题和不忠实的响应方面失败。在本文中，我们提出了一种称为“操作链”的机制，使VLM能够通过一系列的操作来解决问题，其中每个操作都指的是对视觉输入的操作，可以是通过先前训练获得的内在能力（例如，基础）或者是模仿类人行为（例如，放大）。这个机制鼓励VLM生成带有证据的视觉推理的忠实的响应，并允许用户在可解释的路径上追踪错误的原因。因此，我们训练了CogCoM，一个具有内置推理机制的17B通用VLM。实验证明，我们的模型达到了最先进的水平。

    Vision-Language Models (VLMs) have demonstrated their widespread viability thanks to extensive training in aligning visual instructions to answers. However, this conclusive alignment leads models to ignore critical visual reasoning, and further result in failures on meticulous visual problems and unfaithful responses. In this paper, we propose Chain of Manipulations, a mechanism that enables VLMs to solve problems with a series of manipulations, where each manipulation refers to an operation on the visual input, either from intrinsic abilities (e.g., grounding) acquired through prior training or from imitating human-like behaviors (e.g., zoom in). This mechanism encourages VLMs to generate faithful responses with evidential visual reasoning, and permits users to trace error causes in the interpretable paths. We thus train CogCoM, a general 17B VLM with a memory-based compatible architecture endowed this reasoning mechanism. Experiments show that our model achieves the state-of-the-art 
    
[^6]: SoK：面部深度伪造检测器

    SoK: Facial Deepfake Detectors. (arXiv:2401.04364v1 [cs.CV])

    [http://arxiv.org/abs/2401.04364](http://arxiv.org/abs/2401.04364)

    本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。

    

    深度伪造技术迅速成为对社会构成深远和严重威胁的原因之一，主要由于其易于制作和传播。这种情况加速了深度伪造检测技术的发展。然而，许多现有的检测器在验证时 heavily 依赖实验室生成的数据集，这可能无法有效地让它们应对新颖、新兴和实际的深度伪造技术。本文对最新的深度伪造检测器进行广泛全面的回顾和分析，根据几个关键标准对它们进行评估。这些标准将这些检测器分为 4 个高级组别和 13 个细粒度子组别，都遵循一个统一的标准概念框架。这种分类和框架提供了对影响检测器功效的因素的深入和实用的见解。我们对 16 个主要的检测器在各种标准的攻击场景中的普适性进行评估，包括黑盒攻击场景。

    Deepfakes have rapidly emerged as a profound and serious threat to society, primarily due to their ease of creation and dissemination. This situation has triggered an accelerated development of deepfake detection technologies. However, many existing detectors rely heavily on lab-generated datasets for validation, which may not effectively prepare them for novel, emerging, and real-world deepfake techniques. In this paper, we conduct an extensive and comprehensive review and analysis of the latest state-of-the-art deepfake detectors, evaluating them against several critical criteria. These criteria facilitate the categorization of these detectors into 4 high-level groups and 13 fine-grained sub-groups, all aligned with a unified standard conceptual framework. This classification and framework offer deep and practical insights into the factors that affect detector efficacy. We assess the generalizability of 16 leading detectors across various standard attack scenarios, including black-bo
    
[^7]: 通过基于分数的对抗图像生成评估鲁棒性

    Assessing Robustness via Score-Based Adversarial Image Generation. (arXiv:2310.04285v1 [cs.CV])

    [http://arxiv.org/abs/2310.04285](http://arxiv.org/abs/2310.04285)

    本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。

    

    大多数对抗攻击和防御都集中在小的$\ell_p$-范数约束内的扰动上。然而，$\ell_p$威胁模型无法捕捉到所有相关的保留语义的扰动，因此，鲁棒性评估的范围是有限的。在这项工作中，我们引入了基于分数的对抗生成（ScoreAG），一种利用基于分数的生成模型的进展来生成超过$\ell_p$-范数约束的对抗性示例的新的框架，称为无限制的对抗性示例，克服了它们的局限性。与传统方法不同，ScoreAG在生成逼真的对抗性示例时保持图像的核心语义，可以通过转换现有图像或完全从零开始合成新图像的方式实现。我们进一步利用ScoreAG的生成能力来净化图像，从经验上增强分类器的鲁棒性。我们的大量实证评估表明，ScoreAG与现有最先进的对抗攻击方法的性能相当。

    Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantic-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate adversarial examples beyond $\ell_p$-norm constraints, so-called unrestricted adversarial examples, overcoming their limitations. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating realistic adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG matches the performance of state-of-the-art atta
    
[^8]: 表示工程化：AI透明化的自上而下方法

    Representation Engineering: A Top-Down Approach to AI Transparency. (arXiv:2310.01405v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.01405](http://arxiv.org/abs/2310.01405)

    这项研究介绍了一种名为表示工程化（RepE）的自上而下方法，通过借鉴认知神经科学的见解，提供了一种增强AI系统透明性的解决方案。该方法将集群级别的表示放在分析的核心，为监测和操纵深度神经网络中的高级认知现象提供了新的方法，并展示了在解决与安全相关的问题上的潜力。

    

    本文中，我们确定并描述了表示工程化（RepE）这一新兴领域，这是一种通过借鉴认知神经科学的见解来增强AI系统透明性的方法。RepE将集群级别的表示放在分析的核心，而不是神经元或电路，为我们提供了监测和操纵深度神经网络（DNNs）中高级认知现象的新方法。我们提供了RepE技术的基准和初步分析，显示它们提供了简单而有效的解决方案，用于改善我们对大型语言模型的理解和控制。我们展示了这些方法如何在包括诚实性、无害性、追求权力等一系列与安全相关的问题上发挥作用，展示了自上而下透明性研究的潜力。我们希望这项工作能够促进RepE的进一步探索，并推动AI系统的透明性和安全性的进步。

    In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research. We hope that this work catalyzes further exploration of RepE and fosters advancements in the transparency and safety of AI systems.
    
[^9]: 自我监督异常检测的端到端增强超参数调整

    End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection. (arXiv:2306.12033v1 [cs.LG])

    [http://arxiv.org/abs/2306.12033](http://arxiv.org/abs/2306.12033)

    这项研究提出了一种名为ST-SSAD的新方法，可以系统地调整数据增强的超参数，从而有助于提高自我监督异常检测（SSAD）的性能。

    

    自我监督学习（SSL）已经成为一个有前途的范例，它为现实问题提供自产生的监督信号，避免了繁琐的手动标注工作。SSL对于无监督任务，如异常检测尤其具有吸引力，因为标记的异常通常不存在或难以获得。虽然自我监督异常检测（SSAD）近年来受到了广泛关注，但文献却未将数据增强视为超参数。同时，最近的研究表明，增强选择对检测性能有重要影响。在本文中，我们介绍了ST-SSAD（自我调整自我监督异常检测），这是一种关于严格调整增强的SSAD的第一个系统方法。为此，我们的工作提出了两个关键贡献。第一是一种新的无监督验证损失函数，量化增强训练数据与（无标签）测试数据之间的对齐程度。在原则上，我们采用了最近高效的有监督学习方法借鉴的无监督验证方案和增强数据搜索策略，并将其适应于SSAD。我们进一步提出了一种新的增强搜索方法，通过贝叶斯优化的形式，将轻量级数据增强搜索器的简单集成。在各种异常检测基准数据集上的实验表明，我们的增强调整方法相对于以前的最新结果可以获得一致的性能提升，并且相对于最近的有监督方法具有竞争性的结果。

    Self-supervised learning (SSL) has emerged as a promising paradigm that presents self-generated supervisory signals to real-world problems, bypassing the extensive manual labeling burden. SSL is especially attractive for unsupervised tasks such as anomaly detection, where labeled anomalies are often nonexistent and costly to obtain. While self-supervised anomaly detection (SSAD) has seen a recent surge of interest, the literature has failed to treat data augmentation as a hyperparameter. Meanwhile, recent works have reported that the choice of augmentation has significant impact on detection performance. In this paper, we introduce ST-SSAD (Self-Tuning Self-Supervised Anomaly Detection), the first systematic approach to SSAD in regards to rigorously tuning augmentation. To this end, our work presents two key contributions. The first is a new unsupervised validation loss that quantifies the alignment between the augmented training data and the (unlabeled) test data. In principle we adop
    
[^10]: 开放放射组学：一系列标准化数据集和可重复放射组学机器学习流程的技术协议

    Open-radiomics: A Collection of Standardized Datasets and a Technical Protocol for Reproducible Radiomics Machine Learning Pipelines. (arXiv:2207.14776v2 [q-bio.QM] UPDATED)

    [http://arxiv.org/abs/2207.14776](http://arxiv.org/abs/2207.14776)

    本研究提出了一套开放放射组学数据集和技术协议，旨在解决放射组学在结果可重复性和可访问性方面所面临的挑战。通过在BraTS 2020数据集上进行实验，研究了放射组学特征提取对结果可重复性的影响。

    

    目的：作为医学影像中机器学习流程的一个重要分支，放射组学面临着两个主要挑战，即可重复性和可访问性。在这项工作中，我们介绍了开放放射组学，一套放射组学数据集以及基于我们提出的技术协议的综合放射组学流程，以研究放射组学特征提取对结果可重复性的影响。材料和方法：实验使用BraTS 2020开源磁共振成像（MRI）数据集进行，包括369名患有脑肿瘤的成年患者（76例低级别胶质瘤（LGG）和293例高级别胶质瘤（HGG））。使用PyRadiomics库进行LGG与HGG分类，形成了288个放射组学数据集；其中包括4个MRI序列、3个binWidths、6种图像归一化方法和4个肿瘤次区域的组合。使用随机森林分类器，并为每个放射组学数据集进行训练-验证-测试（60%/20%/20%）实验，采用不同的数据划分和m

    Purpose: As an important branch of machine learning pipelines in medical imaging, radiomics faces two major challenges namely reproducibility and accessibility. In this work, we introduce open-radiomics, a set of radiomics datasets along with a comprehensive radiomics pipeline based on our proposed technical protocol to investigate the effects of radiomics feature extraction on the reproducibility of the results.  Materials and Methods: Experiments are conducted on BraTS 2020 open-source Magnetic Resonance Imaging (MRI) dataset that includes 369 adult patients with brain tumors (76 low-grade glioma (LGG), and 293 high-grade glioma (HGG)). Using PyRadiomics library for LGG vs. HGG classification, 288 radiomics datasets are formed; the combinations of 4 MRI sequences, 3 binWidths, 6 image normalization methods, and 4 tumor subregions.  Random Forest classifiers were used, and for each radiomics dataset the training-validation-test (60%/20%/20%) experiment with different data splits and m
    

