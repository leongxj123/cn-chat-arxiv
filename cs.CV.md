# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Autoencoders for Medical Anomaly Detection from A Theoretical Perspective](https://arxiv.org/abs/2403.09303) | 该研究从理论角度为医学异常检测中基于自编码器的重建方法提供了基础，揭示了改进AE在异常检测中的关键在于最小化信息。 |
| [^2] | [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](https://arxiv.org/abs/2403.08551) | 通过2D高斯喷涂实现图像表示和压缩，在GPU内存占用降低的情况下，提供了更快的渲染速度，并在表示性能上与INR相匹敌。 |
| [^3] | [Optimizing Negative Prompts for Enhanced Aesthetics and Fidelity in Text-To-Image Generation](https://arxiv.org/abs/2403.07605) | 提出NegOpt方法，通过监督微调和强化学习优化负面提示的生成，显著提高图像生成质量，超越其他方法并构建了负面提示数据集。 |
| [^4] | [Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning](https://arxiv.org/abs/2403.07362) | 该论文从对抗的角度提出了一种新的机器遗忘评估方法，通过确定最具挑战性的数据子集，即最坏情况遗忘集，来增强对影响擦除的挑战。 |
| [^5] | [AUFormer: Vision Transformers are Parameter-Efficient Facial Action Unit Detectors](https://arxiv.org/abs/2403.04697) | AUFormer提出了一种参数高效的面部动作单位检测方法，引入了新颖的知识混合专家协作机制，解决了传统方法在稀缺数据集或过度依赖额外数据导致的过拟合问题。 |
| [^6] | [Robustness and Exploration of Variational and Machine Learning Approaches to Inverse Problems: An Overview](https://arxiv.org/abs/2402.12072) | 本论文概述了使用变分方法和机器学习解决成像中逆问题的方法，重点在于点估计器对抗性扰动下的鲁棒性以及探索数据一致解子空间以满足特定语义或纹理特性。 |
| [^7] | [Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2401.17263) | 该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。 |
| [^8] | [Machine learning-based analysis of glioma tissue sections: a review.](http://arxiv.org/abs/2401.15022) | 机器学习技术在胶质瘤组织切片分析中具有诊断和预测的潜力，当前研究聚焦于成人型弥漫性胶质瘤的苏木精和伊红染色组织切片，以及对该疾病的分类、分级、分子标记预测和生存预测等临床任务。 |
| [^9] | [Physics-Informed with Power-Enhanced Residual Network for Interpolation and Inverse Problems.](http://arxiv.org/abs/2310.15690) | 本文介绍了一种名为增强型残差网络的新颖神经网络结构，通过在残差元素中添加幂次项提升了网络的表达能力，具有卓越的准确性和应用性能，尤其适用于非平滑函数的处理。同时，该网络结构在解决反问题方面也表现出卓越的性能。 |
| [^10] | [Decoding Human Activities: Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition.](http://arxiv.org/abs/2310.02011) | 本文提出了一种用于活动识别的分层多结构方法，利用残差网络和残差MobileNet对静态和动态活动进行分类，然后通过加权合奏方法进行集成。 |
| [^11] | [CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.](http://arxiv.org/abs/2308.15136) | CAGRA是一种面向GPU的高度并行图构建和近似最近邻搜索方法，在近似最近邻搜索领域取得了显著的效率提升。 |
| [^12] | [Does CLIP Know My Face?.](http://arxiv.org/abs/2209.07341) | 本文提出了一种新方法IDIA来评估视觉语言模型的隐私，大规模实验表明使用于训练的个人可以被非常高的准确率识别出来，表明需要更好地解决视觉语言模型中的隐私问题。 |
| [^13] | [MetaCOG: Learning a Metacognition to Recover What Objects Are Actually There.](http://arxiv.org/abs/2110.03105) | MetaCOG是一个学习元认知的模型，通过学习目标检测器的可靠性表示，增加了目标检测器的鲁棒性，而无需反馈和地面真实的物体标签。 |

# 详细

[^1]: 用理论视角重新思考医学异常检测中的自编码器

    Rethinking Autoencoders for Medical Anomaly Detection from A Theoretical Perspective

    [https://arxiv.org/abs/2403.09303](https://arxiv.org/abs/2403.09303)

    该研究从理论角度为医学异常检测中基于自编码器的重建方法提供了基础，揭示了改进AE在异常检测中的关键在于最小化信息。

    

    医学异常检测旨在仅使用正常训练数据识别异常发现，对健康筛查和识别罕见疾病至关重要。基于重建的方法，特别是利用自编码器（AEs）的方法在这一领域占主导地位。它们基于这样的假设工作：仅使用正常数据训练的AEs不能很好地重建看不见的异常区域，从而实现基于重建错误的异常检测。然而，由于重建训练目标与异常检测任务目标之间的不匹配，这一假设并不总是成立，使得这些方法在理论上不够合理。该研究侧重于为基于AE的重建方法在异常检测中提供理论基础。通过利用信息论，我们阐明了这些方法的原则，并揭示了改进AE在异常检测中的关键在于最小化信息。

    arXiv:2403.09303v1 Announce Type: new  Abstract: Medical anomaly detection aims to identify abnormal findings using only normal training data, playing a crucial role in health screening and recognizing rare diseases. Reconstruction-based methods, particularly those utilizing autoencoders (AEs), are dominant in this field. They work under the assumption that AEs trained on only normal data cannot reconstruct unseen abnormal regions well, thereby enabling the anomaly detection based on reconstruction errors. However, this assumption does not always hold due to the mismatch between the reconstruction training objective and the anomaly detection task objective, rendering these methods theoretically unsound. This study focuses on providing a theoretical foundation for AE-based reconstruction methods in anomaly detection. By leveraging information theory, we elucidate the principles of these methods and reveal that the key to improving AE in anomaly detection lies in minimizing the informati
    
[^2]: 高斯图像：通过2D高斯喷涂进行1000帧每秒的图像表示和压缩

    GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting

    [https://arxiv.org/abs/2403.08551](https://arxiv.org/abs/2403.08551)

    通过2D高斯喷涂实现图像表示和压缩，在GPU内存占用降低的情况下，提供了更快的渲染速度，并在表示性能上与INR相匹敌。

    

    最近，隐式神经表示（INR）在图像表示和压缩方面取得了巨大成功，提供了高视觉质量和快速渲染速度，每秒10-1000帧，假设有足够的GPU资源可用。然而，这种要求常常阻碍了它们在内存有限的低端设备上的使用。为此，我们提出了一种通过2D高斯喷涂进行图像表示和压缩的开创性范式，名为GaussianImage。我们首先引入2D高斯来表示图像，其中每个高斯具有8个参数，包括位置、协方差和颜色。随后，我们揭示了一种基于累积求和的新颖渲染算法。值得注意的是，我们的方法使用GPU内存至少降低3倍，拟合时间快5倍，不仅在表示性能上与INR（例如WIRE，I-NGP）不相上下，而且无论参数大小如何都能提供1500-2000帧每秒的更快渲染速度。

    arXiv:2403.08551v1 Announce Type: cross  Abstract: Implicit neural representations (INRs) recently achieved great success in image representation and compression, offering high visual quality and fast rendering speeds with 10-1000 FPS, assuming sufficient GPU resources are available. However, this requirement often hinders their use on low-end devices with limited memory. In response, we propose a groundbreaking paradigm of image representation and compression by 2D Gaussian Splatting, named GaussianImage. We first introduce 2D Gaussian to represent the image, where each Gaussian has 8 parameters including position, covariance and color. Subsequently, we unveil a novel rendering algorithm based on accumulated summation. Remarkably, our method with a minimum of 3$\times$ lower GPU memory usage and 5$\times$ faster fitting time not only rivals INRs (e.g., WIRE, I-NGP) in representation performance, but also delivers a faster rendering speed of 1500-2000 FPS regardless of parameter size. 
    
[^3]: 优化负面提示以增强文本到图像生成中的美学和保真度

    Optimizing Negative Prompts for Enhanced Aesthetics and Fidelity in Text-To-Image Generation

    [https://arxiv.org/abs/2403.07605](https://arxiv.org/abs/2403.07605)

    提出NegOpt方法，通过监督微调和强化学习优化负面提示的生成，显著提高图像生成质量，超越其他方法并构建了负面提示数据集。

    

    在文本到图像生成中，使用描述不良图像特征的负面提示可以显著提高图像质量。然而，生成良好的负面提示是一项手工而繁琐的工作。为了解决这个问题，我们提出了NegOpt，一种新颖的方法，通过监督微调和强化学习来优化负面提示生成，从而增强图像生成。我们的综合方法相对于其他方法大幅提高了25%的Inception Score，并超越了来自测试集的标准负面提示。此外，使用NegOpt，我们可以有选择地优化对我们最重要的指标。最后，我们构建了负面提示数据集Negative Prompts DB。

    arXiv:2403.07605v1 Announce Type: cross  Abstract: In text-to-image generation, using negative prompts, which describe undesirable image characteristics, can significantly boost image quality. However, producing good negative prompts is manual and tedious. To address this, we propose NegOpt, a novel method for optimizing negative prompt generation toward enhanced image generation, using supervised fine-tuning and reinforcement learning. Our combined approach results in a substantial increase of 25% in Inception Score compared to other approaches and surpasses ground-truth negative prompts from the test set. Furthermore, with NegOpt we can preferentially optimize the metrics most important to us. Finally, we construct Negative Prompts DB, a dataset of negative prompts.
    
[^4]: 挑战遗忘：揭示机器遗忘中最坏情况遗忘集

    Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning

    [https://arxiv.org/abs/2403.07362](https://arxiv.org/abs/2403.07362)

    该论文从对抗的角度提出了一种新的机器遗忘评估方法，通过确定最具挑战性的数据子集，即最坏情况遗忘集，来增强对影响擦除的挑战。

    

    靠谱的机器学习(Machine Learning, ML)社区越来越认识到模型在训练后有选择性地“遗忘”数据点的重要性。这引出了机器遗忘(Machine Unlearning, MU)问题，旨在消除选定数据点对模型性能的影响，同时仍保持模型在遗忘后的实用性。尽管有各种MU方法来擦除数据影响，评估主要集中在随机数据遗忘上，忽视了对于真实衡量遗忘性能的数据子集选择的重要探究。为解决这一问题，我们从对抗的角度引入了一种新的MU评估视角。我们提出确定那些对影响擦除构成最大挑战的数据子集，即找出最坏情况遗忘集。利用双层优化原则，我们增强了在上层优化中的遗忘挑战。

    arXiv:2403.07362v1 Announce Type: cross  Abstract: The trustworthy machine learning (ML) community is increasingly recognizing the crucial need for models capable of selectively 'unlearning' data points after training. This leads to the problem of machine unlearning (MU), aiming to eliminate the influence of chosen data points on model performance, while still maintaining the model's utility post-unlearning. Despite various MU methods for data influence erasure, evaluations have largely focused on random data forgetting, ignoring the vital inquiry into which subset should be chosen to truly gauge the authenticity of unlearning performance. To tackle this issue, we introduce a new evaluative angle for MU from an adversarial viewpoint. We propose identifying the data subset that presents the most significant challenge for influence erasure, i.e., pinpointing the worst-case forget set. Utilizing a bi-level optimization principle, we amplify unlearning challenges at the upper optimization 
    
[^5]: AUFormer: 视觉Transformer是参数高效的面部动作单位检测器

    AUFormer: Vision Transformers are Parameter-Efficient Facial Action Unit Detectors

    [https://arxiv.org/abs/2403.04697](https://arxiv.org/abs/2403.04697)

    AUFormer提出了一种参数高效的面部动作单位检测方法，引入了新颖的知识混合专家协作机制，解决了传统方法在稀缺数据集或过度依赖额外数据导致的过拟合问题。

    

    面部动作单位（AU）在情感计算领域是一个重要概念，AU检测一直是一个热门的研究课题。现有方法由于在稀缺的AU注释数据集上利用大量可学习参数或过度依赖大量额外相关数据而存在过拟合问题。参数高效迁移学习（PETL）提供了一个有希望解决这些挑战的范式，然而其现有方法缺乏针对AU特征的设计。因此，我们创新性地将PETL范式应用于AU检测，引入AUFormer并提出了一种新颖的知识混合专家（MoKE）协作机制。一个特定于某个AU并具有最少可学习参数的MoKE首先集成个性化的多尺度和相关知识。然后MoKE与专家组中的其他MoKE合作，获取聚合信息并将其注入到...

    arXiv:2403.04697v1 Announce Type: cross  Abstract: Facial Action Units (AU) is a vital concept in the realm of affective computing, and AU detection has always been a hot research topic. Existing methods suffer from overfitting issues due to the utilization of a large number of learnable parameters on scarce AU-annotated datasets or heavy reliance on substantial additional relevant data. Parameter-Efficient Transfer Learning (PETL) provides a promising paradigm to address these challenges, whereas its existing methods lack design for AU characteristics. Therefore, we innovatively investigate PETL paradigm to AU detection, introducing AUFormer and proposing a novel Mixture-of-Knowledge Expert (MoKE) collaboration mechanism. An individual MoKE specific to a certain AU with minimal learnable parameters first integrates personalized multi-scale and correlation knowledge. Then the MoKE collaborates with other MoKEs in the expert group to obtain aggregated information and inject it into the 
    
[^6]: 变分方法与机器学习方法在逆问题中的鲁棒性和探索：概述

    Robustness and Exploration of Variational and Machine Learning Approaches to Inverse Problems: An Overview

    [https://arxiv.org/abs/2402.12072](https://arxiv.org/abs/2402.12072)

    本论文概述了使用变分方法和机器学习解决成像中逆问题的方法，重点在于点估计器对抗性扰动下的鲁棒性以及探索数据一致解子空间以满足特定语义或纹理特性。

    

    本文试图概述使用变分方法和机器学习来解决成像中逆问题的当前方法。重点关注点估计器及其对抗性扰动下的鲁棒性。此外，通过一维示例问题的数值实验结果，展示了不同方法的鲁棒性并在经验上验证了理论保证。该综述的另一个重点是通过明确指导来探索数据一致解的子空间，以满足特定语义或纹理特性。

    arXiv:2402.12072v1 Announce Type: cross  Abstract: This paper attempts to provide an overview of current approaches for solving inverse problems in imaging using variational methods and machine learning. A special focus lies on point estimators and their robustness against adversarial perturbations. In this context results of numerical experiments for a one-dimensional toy problem are provided, showing the robustness of different approaches and empirically verifying theoretical guarantees. Another focus of this review is the exploration of the subspace of data consistent solutions through explicit guidance to satisfy specific semantic or textural properties.
    
[^7]: 鲁棒的提示优化用于对抗语言模型的破解攻击

    Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks

    [https://arxiv.org/abs/2401.17263](https://arxiv.org/abs/2401.17263)

    该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。

    

    尽管在人工智能对齐方面取得了一些进展，但语言模型（LM）仍然容易受到对抗性攻击或破解攻击的影响，其中对手修改输入提示以诱导有害行为。虽然已经提出了一些防御方法，但它们仅关注狭窄的威胁模型，并不能提供强大的防御。为了实现强大的防御，我们首次提出了用于对抗破解攻击的对抗目标，并提出了一种名为鲁棒提示优化（RPO）的算法，该算法利用基于梯度的令牌优化来确保输出的无害性。通过这种方法，我们得到了一个易于访问的后缀，显著改善了对破解攻击的强韧性，包括优化过程中出现的破解攻击以及未知的破解攻击，将攻击成功率从84%降低到8.66%，在20个破解攻击中。此外，我们还发现RPO对正常LM使用的影响较小，在适应性攻击下仍然有效，并且可以迁移到黑盒模型中，降低攻击成功率。

    Despite advances in AI alignment, language models (LM) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries modify input prompts to induce harmful behavior. While some defenses have been proposed, they focus on narrow threat models and fall short of a strong defense, which we posit should be effective, universal, and practical. To achieve this, we propose the first adversarial objective for defending LMs against jailbreaking attacks and an algorithm, robust prompt optimization (RPO), that uses gradient-based token optimization to enforce harmless outputs. This results in an easily accessible suffix that significantly improves robustness to both jailbreaks seen during optimization and unknown, held-out jailbreaks, reducing the attack success rate on Starling-7B from 84% to 8.66% across 20 jailbreaks. In addition, we find that RPO has a minor effect on normal LM use, is successful under adaptive attacks, and can transfer to black-box models, reducing the success
    
[^8]: 基于机器学习的胶质瘤组织切片分析：一项综述

    Machine learning-based analysis of glioma tissue sections: a review. (arXiv:2401.15022v1 [eess.IV])

    [http://arxiv.org/abs/2401.15022](http://arxiv.org/abs/2401.15022)

    机器学习技术在胶质瘤组织切片分析中具有诊断和预测的潜力，当前研究聚焦于成人型弥漫性胶质瘤的苏木精和伊红染色组织切片，以及对该疾病的分类、分级、分子标记预测和生存预测等临床任务。

    

    近年来，胶质瘤的诊断变得越来越复杂。使用现代机器学习技术对胶质瘤组织进行组织学评估，为诊断和预测结果提供了新的机会。为了对当前研究的现状进行概述，本综述对70个公开可得的研究论文进行了研究，这些论文关于使用机器学习分析染色的胶质瘤组织切片，涵盖了分类（16/70），分级（23/70），分子标记预测（13/70）和生存预测（27/70）等诊断任务。所有的研究都在方法学方面及其临床适用性方面进行了评估。发现当前研究的重点是对成人型弥漫性胶质瘤的苏木精和伊红染色组织切片进行评估。多数研究（49/70）基于公开的胶质母细胞瘤和低级别胶质瘤数据集，仅有少数研究使用其他数据集。

    In recent years, the diagnosis of gliomas has become increasingly complex. Histological assessment of glioma tissue using modern machine learning techniques offers new opportunities to support diagnosis and outcome prediction. To give an overview of the current state of research, this review examines 70 publicly available research studies on machine learning-based analysis of stained human glioma tissue sections, covering the diagnostic tasks of subtyping (16/70), grading (23/70), molecular marker prediction (13/70), and survival prediction (27/70). All studies were reviewed with regard to methodological aspects as well as clinical applicability. It was found that the focus of current research is the assessment of hematoxylin and eosin-stained tissue sections of adult-type diffuse gliomas. The majority of studies (49/70) are based on the publicly available glioblastoma and low-grade glioma datasets from The Cancer Genome Atlas (TCGA) and only a few studies employed other datasets in is
    
[^9]: 使用增强型残差网络进行插值和反问题的物理驱动方法

    Physics-Informed with Power-Enhanced Residual Network for Interpolation and Inverse Problems. (arXiv:2310.15690v1 [cs.LG])

    [http://arxiv.org/abs/2310.15690](http://arxiv.org/abs/2310.15690)

    本文介绍了一种名为增强型残差网络的新颖神经网络结构，通过在残差元素中添加幂次项提升了网络的表达能力，具有卓越的准确性和应用性能，尤其适用于非平滑函数的处理。同时，该网络结构在解决反问题方面也表现出卓越的性能。

    

    本文介绍了一种新颖的神经网络结构，称为增强型残差网络，旨在改善2D和3D环境下平滑和非平滑函数的插值能力。通过在残差元素中添加幂次项，该网络结构增强了网络的表达能力。研究探究了网络深度、宽度和优化方法，并展示了该网络结构的适应性和性能优势。结果一致表明，增强型残差网络在非平滑函数方面具有异常的准确性。实际示例也证实了其在准确性、收敛性和效率方面相对于普通神经网络的优越性。研究还探讨了更深层网络的影响。此外，提出的网络结构还应用于解决反Burgers方程问题，展示了优越的性能。总之，增强型残差网络提供了一种多功能的解决方案，明显提升了插值和反问题的能力。

    This paper introduces a novel neural network structure called the Power-Enhancing residual network, designed to improve interpolation capabilities for both smooth and non-smooth functions in 2D and 3D settings. By adding power terms to residual elements, the architecture boosts the network's expressive power. The study explores network depth, width, and optimization methods, showing the architecture's adaptability and performance advantages. Consistently, the results emphasize the exceptional accuracy of the proposed Power-Enhancing residual network, particularly for non-smooth functions. Real-world examples also confirm its superiority over plain neural network in terms of accuracy, convergence, and efficiency. The study also looks at the impact of deeper network. Moreover, the proposed architecture is also applied to solving the inverse Burgers' equation, demonstrating superior performance. In conclusion, the Power-Enhancing residual network offers a versatile solution that significa
    
[^10]: 解码人类行为：分析可穿戴加速度计和陀螺仪数据进行活动识别

    Decoding Human Activities: Analyzing Wearable Accelerometer and Gyroscope Data for Activity Recognition. (arXiv:2310.02011v1 [cs.CV])

    [http://arxiv.org/abs/2310.02011](http://arxiv.org/abs/2310.02011)

    本文提出了一种用于活动识别的分层多结构方法，利用残差网络和残差MobileNet对静态和动态活动进行分类，然后通过加权合奏方法进行集成。

    

    一个人的运动或相对定位有效地产生了可以被计算机读取的原始电信号，通过应用各种操作技术来对不同的人类活动进行分类。本文提出了一种基于残差网络与残差MobileNet进行合奏的分层多结构方法，称为FusionActNet。所提出的方法涉及使用精心设计的残差块分别对静态和动态活动进行分类，因为它们具有明显而独特的特征。这些网络独立训练，得到两个专业的高精度模型。通过利用架构调整的算法优势，这些模型在特定超类中优秀地识别活动。然后，这两个残差网络通过加权合奏的残差MobileNet进行传递。随后，这个合奏能够有效区分一些特定的子类。

    A person's movement or relative positioning effectively generates raw electrical signals that can be read by computing machines to apply various manipulative techniques for the classification of different human activities. In this paper, a stratified multi-structural approach based on a Residual network ensembled with Residual MobileNet is proposed, termed as FusionActNet. The proposed method involves using carefully designed Residual blocks for classifying the static and dynamic activities separately because they have clear and distinct characteristics that set them apart. These networks are trained independently, resulting in two specialized and highly accurate models. These models excel at recognizing activities within a specific superclass by taking advantage of the unique algorithmic benefits of architectural adjustments. Afterward, these two ResNets are passed through a weighted ensemble-based Residual MobileNet. Subsequently, this ensemble proficiently discriminates between a sp
    
[^11]: CAGRA：面向GPU的高度并行图构建和近似最近邻搜索

    CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs. (arXiv:2308.15136v1 [cs.DS])

    [http://arxiv.org/abs/2308.15136](http://arxiv.org/abs/2308.15136)

    CAGRA是一种面向GPU的高度并行图构建和近似最近邻搜索方法，在近似最近邻搜索领域取得了显著的效率提升。

    

    近似最近邻搜索（ANNS）在数据挖掘和人工智能领域中起着关键作用，涵盖了信息检索、计算机视觉、自然语言处理和推荐系统等各个学科。近年来，数据量急剧增加，穷举精确最近邻搜索的计算成本往往是禁止性的，必须采用近似技术。尽管图形化方法的平衡性能和召回率在ANNS算法中最近引起了广泛关注，但只有少数研究探索了利用GPU和多核处理器的强大计算能力，尽管广泛使用了大规模并行和通用计算能力。为了弥补这一差距，我们引入了一种基于并行计算硬件的新颖接近图和搜索算法。通过利用现代硬件的高性能能力，我们的方法实现了显著的效率提升。具体而言，我们的方法实现了高效的图构建和近似最近邻搜索。

    Approximate Nearest Neighbor Search (ANNS) plays a critical role in various disciplines spanning data mining and artificial intelligence, from information retrieval and computer vision to natural language processing and recommender systems. Data volumes have soared in recent years and the computational cost of an exhaustive exact nearest neighbor search is often prohibitive, necessitating the adoption of approximate techniques. The balanced performance and recall of graph-based approaches have more recently garnered significant attention in ANNS algorithms, however, only a few studies have explored harnessing the power of GPUs and multi-core processors despite the widespread use of massively parallel and general-purpose computing. To bridge this gap, we introduce a novel parallel computing hardware-based proximity graph and search algorithm. By leveraging the high-performance capabilities of modern hardware, our approach achieves remarkable efficiency gains. In particular, our method s
    
[^12]: CLIP是否知道我的脸？

    Does CLIP Know My Face?. (arXiv:2209.07341v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.07341](http://arxiv.org/abs/2209.07341)

    本文提出了一种新方法IDIA来评估视觉语言模型的隐私，大规模实验表明使用于训练的个人可以被非常高的准确率识别出来，表明需要更好地解决视觉语言模型中的隐私问题。

    

    随着深度学习在各个应用中的普及，保护训练数据的隐私问题已经成为一个关键的研究领域。以前的研究主要关注单模型的隐私风险，我们提出了一种新的方法来评估多模型的隐私，特别是像CLIP这样的视觉语言模型。所提出的身份推断攻击(IDIA)通过用同一人的图片向模型查询，从而揭示该个人是否被包含在训练数据中。让模型从各种可能的文本标签中选择，模型会透露是否识别该人物，从而表明其被用于训练。我们在CLIP上进行的大规模实验表明，使用于训练的个人可以被非常高的准确率识别出来。我们确认该模型已经学会将名称与描绘的个人相关联，这意味着敏感信息存在于其中，可以被对手提取。我们的结果凸显了需要在视觉语言模型中更好地解决隐私问题。

    With the rise of deep learning in various applications, privacy concerns around the protection of training data has become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for 
    
[^13]: MetaCOG: 学习元认知以恢复实际存在的物体

    MetaCOG: Learning a Metacognition to Recover What Objects Are Actually There. (arXiv:2110.03105v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2110.03105](http://arxiv.org/abs/2110.03105)

    MetaCOG是一个学习元认知的模型，通过学习目标检测器的可靠性表示，增加了目标检测器的鲁棒性，而无需反馈和地面真实的物体标签。

    

    人类不仅根据我们所看到的内容形成关于世界的表征，还学习关于我们自己视觉如何工作的元认知表征。这使我们能够识别出我们的视觉不可靠（例如，当我们意识到我们正在经历视觉错觉时），并使我们能够对我们所看到的内容提出质疑。受到这种人类能力的启发，我们提出了MetaCOG：一种通过学习其可靠性表示来增加目标检测器的鲁棒性的模型，并且在没有反馈的情况下实现。具体而言，MetaCOG是一个层次概率模型，对一个三维场景中的物体和检测器产生的输出表达了一个联合分布。当与现成的目标检测器配对使用时，MetaCOG将检测结果作为输入，并推断出检测器错漏检某些类别的物体和虚构不存在的物体的倾向，而无需访问地面真实的物体标签。

    Humans not only form representations about the world based on what we see, but also learn meta-cognitive representations about how our own vision works. This enables us to recognize when our vision is unreliable (e.g., when we realize that we are experiencing a visual illusion) and enables us to question what we see. Inspired by this human capacity, we present MetaCOG: a model that increases the robustness of object detectors by learning representations of their reliability, and does so without feedback. Specifically, MetaCOG is a hierarchical probabilistic model that expresses a joint distribution over the objects in a 3D scene and the outputs produced by a detector. When paired with an off-the-shelf object detector, MetaCOG takes detections as input and infers the detector's tendencies to miss objects of certain categories and to hallucinate objects that are not actually present, all without access to ground-truth object labels. When paired with three modern neural object detectors, 
    

