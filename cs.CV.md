# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predicting risk of cardiovascular disease using retinal OCT imaging](https://arxiv.org/abs/2403.18873) | 这项研究探讨了使用光学相干断层扫描（OCT）作为额外成像技术来预测心血管疾病的潜力，并通过自监督深度学习和随机森林分类器结合的方法成功区分了心血管疾病风险和非风险患者。 |
| [^2] | [Approximation and bounding techniques for the Fisher-Rao distances](https://arxiv.org/abs/2403.10089) | 本文考虑了几种数值上稳健的Fisher-Rao距离的近似和界定技术，包括基于闭合形式1D子模型Fisher-Rao距离的通用上界以及取决于测地线或预测测地线是否闭合形式获得的几种通用近似方案，并提出了一种通用方法保证近似误差任意小。 |
| [^3] | [Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models](https://arxiv.org/abs/2402.16315) | Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。 |
| [^4] | [Multi: Multimodal Understanding Leaderboard with Text and Images](https://arxiv.org/abs/2402.03173) | Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。 |
| [^5] | [GUPNet++: Geometry Uncertainty Propagation Network for Monocular 3D Object Detection.](http://arxiv.org/abs/2310.15624) | GUPNet++是一种通过以概率方式建模几何投影的几何不确定性传播网络，可以提高单目三维物体检测的深度预测稳定性和效率。 |

# 详细

[^1]: 使用视网膜OCT成像预测心血管疾病风险

    Predicting risk of cardiovascular disease using retinal OCT imaging

    [https://arxiv.org/abs/2403.18873](https://arxiv.org/abs/2403.18873)

    这项研究探讨了使用光学相干断层扫描（OCT）作为额外成像技术来预测心血管疾病的潜力，并通过自监督深度学习和随机森林分类器结合的方法成功区分了心血管疾病风险和非风险患者。

    

    我们调查了光学相干断层扫描（OCT）作为一种额外成像技术来预测未来心血管疾病（CVD）的潜力。我们利用基于变分自动编码器（VAE）的自监督深度学习方法学习了高维3D OCT图像的低维表示，并捕捉了OCT图像中不同视网膜层的独特特征。随后使用学习到的潜在特征和参与者的人口统计数据以及临床数据训练了一个随机森林（RF）分类器，以区分处于CVD事件风险（心梗或中风）和非CVD病例的患者。我们的预测模型基于对多模态数据的训练，评估其能力来正确识别在图像获取后的5年内可能患有CVD事件（心梗或中风）的个体。我们的自监督VAE特征选择和多模态随机森林分类器区分

    arXiv:2403.18873v1 Announce Type: cross  Abstract: We investigated the potential of optical coherence tomography (OCT) as an additional imaging technique to predict future cardiovascular disease (CVD). We utilised a self-supervised deep learning approach based on Variational Autoencoders (VAE) to learn low-dimensional representations of high-dimensional 3D OCT images and to capture distinct characteristics of different retinal layers within the OCT image. A Random Forest (RF) classifier was subsequently trained using the learned latent features and participant demographic and clinical data, to differentiate between patients at risk of CVD events (MI or stroke) and non-CVD cases. Our predictive model, trained on multimodal data, was assessed based on its ability to correctly identify individuals likely to suffer from a CVD event(MI or stroke), within a 5-year interval after image acquisition. Our self-supervised VAE feature selection and multimodal Random Forest classifier differentiate
    
[^2]: 用于Fisher-Rao距离的近似和界定技术

    Approximation and bounding techniques for the Fisher-Rao distances

    [https://arxiv.org/abs/2403.10089](https://arxiv.org/abs/2403.10089)

    本文考虑了几种数值上稳健的Fisher-Rao距离的近似和界定技术，包括基于闭合形式1D子模型Fisher-Rao距离的通用上界以及取决于测地线或预测测地线是否闭合形式获得的几种通用近似方案，并提出了一种通用方法保证近似误差任意小。

    

    统计模型的两个概率分布之间的Fisher-Rao距离被定义为Fisher信息度量诱导的Riemannian测地距离。为了以闭合形式计算Fisher-Rao距离，我们需要（1）推导出Fisher-Rao测地线的公式，以及（2）沿着这些测地线积分Fisher长度元素。我们考虑了几种数值上稳健的Fisher-Rao距离的近似和界定技术：首先，我们基于子模型的闭合形式1D Fisher-Rao距离报告了Fisher-Rao距离的通用上界。其次，我们描述了几种通用的近似方案，取决于Fisher-Rao测地线或预测测地线是否能以闭合形式获得。特别地，我们获得了一种通用的方法，可以保证在提供Fisher-Rao预测测地线和严格的下界和上界时近似产生任意小的附加误差。

    arXiv:2403.10089v1 Announce Type: cross  Abstract: The Fisher-Rao distance between two probability distributions of a statistical model is defined as the Riemannian geodesic distance induced by the Fisher information metric. In order to calculate the Fisher-Rao distance in closed-form, we need (1) to elicit a formula for the Fisher-Rao geodesics, and (2) to integrate the Fisher length element along those geodesics. We consider several numerically robust approximation and bounding techniques for the Fisher-Rao distances: First, we report generic upper bounds on Fisher-Rao distances based on closed-form 1D Fisher-Rao distances of submodels. Second, we describe several generic approximation schemes depending on whether the Fisher-Rao geodesics or pregeodesics are available in closed-form or not. In particular, we obtain a generic method to guarantee an arbitrarily small additive error on the approximation provided that Fisher-Rao pregeodesics and tight lower and upper bounds are available
    
[^3]: Finer: 在大型视觉语言模型中研究和增强细粒度视觉概念识别

    Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models

    [https://arxiv.org/abs/2402.16315](https://arxiv.org/abs/2402.16315)

    Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。

    

    最近指导调整的大型视觉语言模型（LVLMs）的进展使模型能够轻松生成高水平的基于图像的解释。尽管这种能力主要归因于大型语言模型（LLMs）中包含的丰富世界知识，但我们的工作揭示了它们在六个不同基准设置下的细粒度视觉分类（FGVC）上的缺陷。最近的LVLMs最先进的模型，如LLaVa-1.5，InstructBLIP和GPT-4V，在分类性能方面严重下降，例如，LLaVA-1.5在斯坦福狗的EM平均下降了65.58，而且还难以根据出现在输入图像中的概念生成具有详细属性的准确解释，尽管它们有生成整体图像级描述的能力。深入分析表明，经过指导调整的LVLMs在给定文本时呈现出模态差距，显示出存在不一致性

    arXiv:2402.16315v1 Announce Type: cross  Abstract: Recent advances in instruction-tuned Large Vision-Language Models (LVLMs) have imbued the models with the ability to generate high-level, image-grounded explanations with ease. While such capability is largely attributed to the rich world knowledge contained within the Large Language Models (LLMs), our work reveals their shortcomings in fine-grained visual categorization (FGVC) across six different benchmark settings. Most recent state-of-the-art LVLMs like LLaVa-1.5, InstructBLIP and GPT-4V not only severely deteriorate in terms of classification performance, e.g., average drop of 65.58 in EM for Stanford Dogs for LLaVA-1.5, but also struggle to generate an accurate explanation with detailed attributes based on the concept that appears within an input image despite their capability to generate holistic image-level descriptions. In-depth analyses show that instruction-tuned LVLMs exhibit modality gap, showing discrepancy when given tex
    
[^4]: 多模态：文本和图像的多模态理解排行榜

    Multi: Multimodal Understanding Leaderboard with Text and Images

    [https://arxiv.org/abs/2402.03173](https://arxiv.org/abs/2402.03173)

    Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。

    

    多模态大型语言模型（MLLM）的快速进展强调了向学术界引入具有挑战性而又真实的基准的需求。现有的基准主要关注简单的自然图像理解，但Multi成为了MLLM的尖端基准，提供了一个综合性的数据集，用于评估MLLM对理解复杂图表和科学问题的能力。该基准反映了当前真实的考试风格，提供多模态的输入，并要求准确或开放式的回答，类似于现实中的学校考试。它通过各种任务挑战MLLM，从公式推导到图像细节分析，以及跨模态推理。Multi包括超过18,000个问题，重点关注不同格式的基于科学的问答。我们还引入了Multi-Elite，一个包含500个问题的子集，用于测试MLLM的极端情况，以及Multi-Extend，通过超过4..。

    Rapid progress in multimodal large language models (MLLMs) highlights the need to introduce challenging yet realistic benchmarks to the academic community. Existing benchmarks primarily focus on simple natural image understanding, but Multi emerges as a cutting-edge benchmark for MLLMs, offering a comprehensive dataset for evaluating MLLMs against understanding complex figures and tables, and scientific questions. This benchmark, reflecting current realistic examination styles, provides multimodal inputs and requires responses that are either precise or open-ended, similar to real-life school tests. It challenges MLLMs with a variety of tasks, ranging from formula derivation to image detail analysis, and cross-modality reasoning. Multi includes over 18,000 questions, with a focus on science-based QA in diverse formats. We also introduce Multi-Elite, a 500-question subset for testing the extremities of MLLMs, and Multi-Extend, which enhances In-Context Learning research with more than 4
    
[^5]: GUPNet++：用于单目三维物体检测的几何不确定性传播网络

    GUPNet++: Geometry Uncertainty Propagation Network for Monocular 3D Object Detection. (arXiv:2310.15624v1 [cs.CV])

    [http://arxiv.org/abs/2310.15624](http://arxiv.org/abs/2310.15624)

    GUPNet++是一种通过以概率方式建模几何投影的几何不确定性传播网络，可以提高单目三维物体检测的深度预测稳定性和效率。

    

    几何在单目三维物体检测中起着重要作用。它可以通过物体的物理尺寸与图像平面中的二维投影之间的透视投影来估计物体的深度，这可以将数学先验引入深度模型。然而，这个投影过程也会引入误差放大，估计高度的误差会被放大并反映到投影的深度中。这导致深度推断不可靠，并且影响训练的稳定性。为了解决这个问题，我们提出了一种新颖的几何不确定性传播网络(GUPNet++)，通过以概率方式建模几何投影。这确保了深度预测是有界的，并与合理的不确定性相关联。引入这种几何不确定性的意义有两个方面：(1)。它模拟了几何投影在训练过程中的不确定性传播关系，提高了端到端模型学习的稳定性和效率。

    Geometry plays a significant role in monocular 3D object detection. It can be used to estimate object depth by using the perspective projection between object's physical size and 2D projection in the image plane, which can introduce mathematical priors into deep models. However, this projection process also introduces error amplification, where the error of the estimated height is amplified and reflected into the projected depth. It leads to unreliable depth inferences and also impairs training stability. To tackle this problem, we propose a novel Geometry Uncertainty Propagation Network (GUPNet++) by modeling geometry projection in a probabilistic manner. This ensures depth predictions are well-bounded and associated with a reasonable uncertainty. The significance of introducing such geometric uncertainty is two-fold: (1). It models the uncertainty propagation relationship of the geometry projection during training, improving the stability and efficiency of the end-to-end model learni
    

