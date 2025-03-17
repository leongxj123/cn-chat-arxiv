# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions](https://arxiv.org/abs/2403.17064) | 通过识别CLIP文本嵌入中的语义方向，实现了文本到图像模型中对高级属性的细粒度主题特定控制。 |
| [^2] | [LIX: Implicitly Infusing Spatial Geometric Prior Knowledge into Visual Semantic Segmentation for Autonomous Driving](https://arxiv.org/abs/2403.08215) | 将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型，通过新的logit蒸馏和特征蒸馏方法，解决自动驾驶中的视觉语义分割问题。 |
| [^3] | [Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement](https://arxiv.org/abs/2403.03551) | 提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。 |
| [^4] | [Masked LoGoNet: Fast and Accurate 3D Image Analysis for Medical Domain](https://arxiv.org/abs/2402.06190) | 本文介绍了一种名为LoGoNet的新型神经网络架构，采用自监督学习方法来应对医学图像分析中的挑战。LoGoNet通过采用大内核注意力和双重编码策略，灵活捕捉长、短距离特征相关性。这种创新的组合技术在医学图像分割中特别有益。 |
| [^5] | [Towards Few-Call Model Stealing via Active Self-Paced Knowledge Distillation and Diffusion-Based Image Generation.](http://arxiv.org/abs/2310.00096) | 本研究提出了一种利用扩散模型生成合成数据集，并通过少次调用的方法窃取黑盒模型的框架，突破了访问限制条件。 |
| [^6] | [In Defense of Pure 16-bit Floating-Point Neural Networks.](http://arxiv.org/abs/2305.10947) | 本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。 |
| [^7] | [Virtual Guidance as a Mid-level Representation for Navigation.](http://arxiv.org/abs/2303.02731) | 该论文介绍了一种名为“虚拟导航”的新技术，通过在智能体的相机视图上叠加彩色路径或球的形式的视觉指引，以易于理解的导航指令传达抽象的导航信息。实验结果表明，在模拟和真实环境中，虚拟导航在遵循计划路径和避开障碍物等多个指标上优于现有方法。 |

# 详细

[^1]: 在T2I模型中通过识别语义方向实现连续、主题特定的属性控制

    Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions

    [https://arxiv.org/abs/2403.17064](https://arxiv.org/abs/2403.17064)

    通过识别CLIP文本嵌入中的语义方向，实现了文本到图像模型中对高级属性的细粒度主题特定控制。

    

    近年来，文本到图像（T2I）扩散模型的进展显著提高了生成图像的质量。然而，由于自然语言提示的限制（例如“人”和“老年人”之间不存在连续的中间描述的集合），实现对属性的细粒度控制仍然是一个挑战。尽管引入了许多方法来增强模型或生成过程以实现这种控制，但不需要固定参考图像的方法仅限于启用全局细粒度属性表达控制或仅限于特定主题的粗粒度属性表达控制，而不能同时兼顾两者。我们展示了在常用的基于标记级别的CLIP文本嵌入中存在可实现文本到图像模型中高级属性的细粒度主题特定控制的方向。基于这一观察，我们引入了一种有效的方法。

    arXiv:2403.17064v1 Announce Type: cross  Abstract: In recent years, advances in text-to-image (T2I) diffusion models have substantially elevated the quality of their generated images. However, achieving fine-grained control over attributes remains a challenge due to the limitations of natural language prompts (such as no continuous set of intermediate descriptions existing between ``person'' and ``old person''). Even though many methods were introduced that augment the model or generation process to enable such control, methods that do not require a fixed reference image are limited to either enabling global fine-grained attribute expression control or coarse attribute expression control localized to specific subjects, not both simultaneously. We show that there exist directions in the commonly used token-level CLIP text embeddings that enable fine-grained subject-specific control of high-level attributes in text-to-image models. Based on this observation, we introduce one efficient op
    
[^2]: LIX：将空间几何先验知识隐式注入视觉语义分割，用于自动驾驶

    LIX: Implicitly Infusing Spatial Geometric Prior Knowledge into Visual Semantic Segmentation for Autonomous Driving

    [https://arxiv.org/abs/2403.08215](https://arxiv.org/abs/2403.08215)

    将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型，通过新的logit蒸馏和特征蒸馏方法，解决自动驾驶中的视觉语义分割问题。

    

    尽管数据融合网络在视觉语义分割中表现出色，但当缺乏空间几何数据时，双编码器变得无效。将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型是一个实用但不太探索的研究领域。本文深入探讨了这个主题，并采用知识蒸馏方法来解决这个问题。我们引入了Learning to Infuse "X" (LIX) 框架，在logit蒸馏和特征蒸馏方面进行了新颖贡献。我们提出了一个数学证明，强调在解耦知识蒸馏中使用单一固定权重的局限性，并引入了logit智能动态权重控制器作为解决这个问题的方法。此外，我们开发了一种自适应重新校准的特征蒸馏算法，包括两种技术。

    arXiv:2403.08215v1 Announce Type: cross  Abstract: Despite the impressive performance achieved by data-fusion networks with duplex encoders for visual semantic segmentation, they become ineffective when spatial geometric data are not available. Implicitly infusing the spatial geometric prior knowledge acquired by a duplex-encoder teacher model into a single-encoder student model is a practical, albeit less explored research avenue. This paper delves into this topic and resorts to knowledge distillation approaches to address this problem. We introduce the Learning to Infuse "X" (LIX) framework, with novel contributions in both logit distillation and feature distillation aspects. We present a mathematical proof that underscores the limitation of using a single fixed weight in decoupled knowledge distillation and introduce a logit-wise dynamic weight controller as a solution to this issue. Furthermore, we develop an adaptively-recalibrated feature distillation algorithm, including two tec
    
[^3]: 通过微调预先为高斯降噪而训练的UNet进行低剂量CT图像重建，用于图像增强的下游任务

    Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement

    [https://arxiv.org/abs/2403.03551](https://arxiv.org/abs/2403.03551)

    提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。

    

    计算机断层扫描（CT）是一种广泛使用的医学成像模态，由于其基于电离辐射，因此希望尽量减少辐射剂量。然而，降低辐射剂量会导致图像质量下降，从低剂量CT（LDCT）数据重建仍然是一个具有挑战性的任务，值得进行研究。根据LoDoPaB-CT基准，许多最先进的方法使用涉及UNet型架构的流程。具体来说，排名第一的方法ItNet使用包括滤波反投影（FBP）、在CT数据上训练的UNet和迭代细化步骤的三阶段流程。在本文中，我们提出了一种更简单的两阶段方法。第一阶段也使用了FBP，而新颖之处在于第二阶段的训练策略，特点是CT图像增强阶段。我们方法的关键点在于神经网络是预训练的。

    arXiv:2403.03551v1 Announce Type: cross  Abstract: Computed Tomography (CT) is a widely used medical imaging modality, and as it is based on ionizing radiation, it is desirable to minimize the radiation dose. However, a reduced radiation dose comes with reduced image quality, and reconstruction from low-dose CT (LDCT) data is still a challenging task which is subject to research. According to the LoDoPaB-CT benchmark, a benchmark for LDCT reconstruction, many state-of-the-art methods use pipelines involving UNet-type architectures. Specifically the top ranking method, ItNet, employs a three-stage process involving filtered backprojection (FBP), a UNet trained on CT data, and an iterative refinement step. In this paper, we propose a less complex two-stage method. The first stage also employs FBP, while the novelty lies in the training strategy for the second stage, characterized as the CT image enhancement stage. The crucial point of our approach is that the neural network is pretrained
    
[^4]: Masked LoGoNet：用于医学领域的快速准确3D图像分析

    Masked LoGoNet: Fast and Accurate 3D Image Analysis for Medical Domain

    [https://arxiv.org/abs/2402.06190](https://arxiv.org/abs/2402.06190)

    本文介绍了一种名为LoGoNet的新型神经网络架构，采用自监督学习方法来应对医学图像分析中的挑战。LoGoNet通过采用大内核注意力和双重编码策略，灵活捕捉长、短距离特征相关性。这种创新的组合技术在医学图像分割中特别有益。

    

    标准的现代机器学习图像方法在医学应用中面临挑战，因为数据集构建的高成本和有限的标记训练数据。此外，这些方法在部署时通常用于每天处理大量数据，给医疗设施带来高维护成本。在本文中，我们引入了一种新的神经网络架构LoGoNet，采用定制的自监督学习（SSL）方法来缓解这些挑战。LoGoNet在U形架构内整合了一种新颖的特征提取器，利用大内核注意力（LKA）和双重编码策略，灵活地捕捉长、短距离特征相关性。这与现有方法依赖增加网络容量以增强特征提取的方式形成对比。我们模型中这些新技术的组合在医学图像分割中特别有益，考虑到其困难性。

    Standard modern machine-learning-based imaging methods have faced challenges in medical applications due to the high cost of dataset construction and, thereby, the limited labeled training data available. Additionally, upon deployment, these methods are usually used to process a large volume of data on a daily basis, imposing a high maintenance cost on medical facilities. In this paper, we introduce a new neural network architecture, termed LoGoNet, with a tailored self-supervised learning (SSL) method to mitigate such challenges. LoGoNet integrates a novel feature extractor within a U-shaped architecture, leveraging Large Kernel Attention (LKA) and a dual encoding strategy to capture both long-range and short-range feature dependencies adeptly. This is in contrast to existing methods that rely on increasing network capacity to enhance feature extraction. This combination of novel techniques in our model is especially beneficial in medical image segmentation, given the difficulty of le
    
[^5]: 近似黑盒模型窃取的少次调用方法：活跃自适应知识蒸馏和基于扩散的图像生成

    Towards Few-Call Model Stealing via Active Self-Paced Knowledge Distillation and Diffusion-Based Image Generation. (arXiv:2310.00096v1 [cs.CV])

    [http://arxiv.org/abs/2310.00096](http://arxiv.org/abs/2310.00096)

    本研究提出了一种利用扩散模型生成合成数据集，并通过少次调用的方法窃取黑盒模型的框架，突破了访问限制条件。

    

    扩散模型在图像合成方面展示了强大的能力，并在许多计算机视觉任务中取得了巨大成功。为此，我们提出了一个新的用例，即在没有访问原始训练数据、架构和模型权重的情况下复制黑盒分类模型，即只能通过推理API使用模型。具体来说，我们只能观察到一些图像样本作为输入传递给模型时的（软性或硬性）标签。此外，我们考虑到限制模型调用次数的额外约束，主要关注于少次调用的模型窃取。为了在应用限制条件的情况下解决模型提取任务，我们提出了以下框架。作为训练数据，我们利用扩散模型生成逼真且多样化的图像创建了一个合成数据集（称为代理数据集）。给定允许的最大API调用次数，我们传递相应数量的样本进行训练。

    Diffusion models showcased strong capabilities in image synthesis, being used in many computer vision tasks with great success. To this end, we propose to explore a new use case, namely to copy black-box classification models without having access to the original training data, the architecture, and the weights of the model, \ie~the model is only exposed through an inference API. More specifically, we can only observe the (soft or hard) labels for some image samples passed as input to the model. Furthermore, we consider an additional constraint limiting the number of model calls, mostly focusing our research on few-call model stealing. In order to solve the model extraction task given the applied restrictions, we propose the following framework. As training data, we create a synthetic data set (called proxy data set) by leveraging the ability of diffusion models to generate realistic and diverse images. Given a maximum number of allowed API calls, we pass the respective number of sampl
    
[^6]: 关于纯16位浮点神经网络的辩护

    In Defense of Pure 16-bit Floating-Point Neural Networks. (arXiv:2305.10947v1 [cs.LG])

    [http://arxiv.org/abs/2305.10947](http://arxiv.org/abs/2305.10947)

    本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。

    

    减少编码神经网络权重和激活所需的位数是非常可取的，因为它可以加快神经网络的训练和推理时间，同时减少内存消耗。因此，这一领域的研究引起了广泛关注，以开发利用更低精度计算的神经网络，比如混合精度训练。有趣的是，目前不存在纯16位浮点设置的方法。本文揭示了纯16位浮点神经网络被忽视的效率。我们通过提供全面的理论分析来探讨造成16位和32位模型的差异的因素。我们规范化了浮点误差和容忍度的概念，从而可以定量解释16位模型与其32位对应物之间密切逼近结果的条件。这种理论探索提供了新的视角。

    Reducing the number of bits needed to encode the weights and activations of neural networks is highly desirable as it speeds up their training and inference time while reducing memory consumption. For these reasons, research in this area has attracted significant attention toward developing neural networks that leverage lower-precision computing, such as mixed-precision training. Interestingly, none of the existing approaches has investigated pure 16-bit floating-point settings. In this paper, we shed light on the overlooked efficiency of pure 16-bit floating-point neural networks. As such, we provide a comprehensive theoretical analysis to investigate the factors contributing to the differences observed between 16-bit and 32-bit models. We formalize the concepts of floating-point error and tolerance, enabling us to quantitatively explain the conditions under which a 16-bit model can closely approximate the results of its 32-bit counterpart. This theoretical exploration offers perspect
    
[^7]: 导航的中层表示——虚拟导航

    Virtual Guidance as a Mid-level Representation for Navigation. (arXiv:2303.02731v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02731](http://arxiv.org/abs/2303.02731)

    该论文介绍了一种名为“虚拟导航”的新技术，通过在智能体的相机视图上叠加彩色路径或球的形式的视觉指引，以易于理解的导航指令传达抽象的导航信息。实验结果表明，在模拟和真实环境中，虚拟导航在遵循计划路径和避开障碍物等多个指标上优于现有方法。

    

    在自主导航的背景下，有效地传达抽象的导航指引给动态环境中的智能体存在挑战，特别是当导航信息是多模态的时候。为了解决这个问题，本文引入了一种名为“虚拟导航”的新技术，旨在以视觉方式呈现非视觉指令信号。这些视觉指引以彩色路径或球的形式叠加在智能体的相机视图上，作为易于理解的导航指令。我们通过在模拟和真实环境中进行实验来评估我们提出的方法。在模拟环境中，我们的虚拟导航在多项指标上优于基线混合方法，包括遵循计划路径和避开障碍物。此外，我们将虚拟导航的概念扩展到将基于文本提示的指令转换为用于真实环境实验的直观视觉格式。我们的结果验证了虚拟导航的适应性。

    In the context of autonomous navigation, effectively conveying abstract navigational cues to agents in dynamic environments poses challenges, particularly when the navigation information is multimodal. To address this issue, the paper introduces a novel technique termed "Virtual Guidance," which is designed to visually represent non-visual instructional signals. These visual cues, rendered as colored paths or spheres, are overlaid onto the agent's camera view, serving as easily comprehensible navigational instructions. We evaluate our proposed method through experiments in both simulated and real-world settings. In the simulated environments, our virtual guidance outperforms baseline hybrid approaches in several metrics, including adherence to planned routes and obstacle avoidance. Furthermore, we extend the concept of virtual guidance to transform text-prompt-based instructions into a visually intuitive format for real-world experiments. Our results validate the adaptability of virtua
    

