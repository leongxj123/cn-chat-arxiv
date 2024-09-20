# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cross--domain Fiber Cluster Shape Analysis for Language Performance Cognitive Score Prediction](https://arxiv.org/abs/2403.19001) | 本研究通过新颖的框架SFFormer，结合了多头交叉注意力特征融合模块，基于dMRI纤维束追踪，预测了主观语言表现，拓展了脑结构与人类认知功能的关联研究。 |
| [^2] | [Avoiding Catastrophic Forgetting in Visual Classification Using Human Concept Formation](https://arxiv.org/abs/2402.16933) | 提出了一种名为Cobweb4V的新颖视觉分类方法，利用人类类似学习系统，避免了灾难性遗忘效应，与传统方法相比，需要更少的数据来实现有效学习成果，并保持稳定性能。 |
| [^3] | [Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation](https://arxiv.org/abs/2402.07127) |  |
| [^4] | [NN-Copula-CD: A Copula-Guided Interpretable Neural Network for Change Detection in Heterogeneous Remote Sensing Images.](http://arxiv.org/abs/2303.17448) | 该论文提出了一种可解释的神经网络方法，结合Copula理论来解决异构遥感图像中的变化检测问题。 |
| [^5] | [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention.](http://arxiv.org/abs/2303.16199) | 本文提出了一种基于适应提示和零初始化注意力机制的轻量级语言模型调整方法，可高效微调LLaMA为指令跟随模型，具有比Alpaca更短的微调时间并具有近似的响应质量。 |
| [^6] | [MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks.](http://arxiv.org/abs/2212.08158) | 该论文提出了一种性能不可知的多模态得分方法MM-SHAP，可以可靠地量化多模态模型使用各自模态的比例，并应用于比较模型的平均多模态程度和衡量个体模型的贡献。实验结果表明单模态崩溃比以前认为的更为普遍，而MM-SHAP是分析VL模型多模态行为的有效工具。 |

# 详细

[^1]: 跨领域的纤维簇形状分析用于语言表现认知分数预测

    Cross--domain Fiber Cluster Shape Analysis for Language Performance Cognitive Score Prediction

    [https://arxiv.org/abs/2403.19001](https://arxiv.org/abs/2403.19001)

    本研究通过新颖的框架SFFormer，结合了多头交叉注意力特征融合模块，基于dMRI纤维束追踪，预测了主观语言表现，拓展了脑结构与人类认知功能的关联研究。

    

    形状在计算机图形学中扮演重要角色，提供了有关对象形态和功能的信息特征。脑成像中的形状分析可帮助解释人脑结构和功能的相关性。本研究调查了大脑的3D白质连接的形状及其与人类认知功能的潜在预测关系。我们使用扩散磁共振成像（dMRI）纤维束追踪将大脑连接重建为3D点序列。为了描述每个连接，我们提取了12个形状描述符以及传统的dMRI连接和组织微结构特征。我们引入了一种新颖的框架，形状融合纤维簇变换器（SFFormer），利用多头交叉注意力特征融合模块基于dMRI纤维束追踪来预测特定个体的语言表现。我们在一个大型数据集上评估了该方法的性能。

    arXiv:2403.19001v1 Announce Type: cross  Abstract: Shape plays an important role in computer graphics, offering informative features to convey an object's morphology and functionality. Shape analysis in brain imaging can help interpret structural and functionality correlations of the human brain. In this work, we investigate the shape of the brain's 3D white matter connections and its potential predictive relationship to human cognitive function. We reconstruct brain connections as sequences of 3D points using diffusion magnetic resonance imaging (dMRI) tractography. To describe each connection, we extract 12 shape descriptors in addition to traditional dMRI connectivity and tissue microstructure features. We introduce a novel framework, Shape--fused Fiber Cluster Transformer (SFFormer), that leverages a multi-head cross-attention feature fusion module to predict subject-specific language performance based on dMRI tractography. We assess the performance of the method on a large dataset
    
[^2]: 使用人类概念形成避免视觉分类中的灾难性遗忘

    Avoiding Catastrophic Forgetting in Visual Classification Using Human Concept Formation

    [https://arxiv.org/abs/2402.16933](https://arxiv.org/abs/2402.16933)

    提出了一种名为Cobweb4V的新颖视觉分类方法，利用人类类似学习系统，避免了灾难性遗忘效应，与传统方法相比，需要更少的数据来实现有效学习成果，并保持稳定性能。

    

    深度神经网络在机器学习中表现出色，特别是在视觉任务中，然而，当按顺序学习新任务时，它们经常面临灾难性遗忘。本研究提出了Cobweb4V，这是一种新颖的视觉分类方法，它基于Cobweb，这是一种人类类似的学习系统，受到人类随时间逐渐学习新概念的启发。我们进行了全面评估，展示了Cobweb4V在学习视觉概念方面的熟练程度，相较于传统方法，需要更少的数据来实现有效的学习成果，随时间保持稳定的性能，并实现了令人称赞的渐近行为，避免了灾难性遗忘效应。这些特征与人类认知中的学习策略一致，将Cobweb4V定位为神经网络方法的一个有前途的替代方案。

    arXiv:2402.16933v1 Announce Type: cross  Abstract: Deep neural networks have excelled in machine learning, particularly in vision tasks, however, they often suffer from catastrophic forgetting when learning new tasks sequentially. In this work, we propose Cobweb4V, a novel visual classification approach that builds on Cobweb, a human like learning system that is inspired by the way humans incrementally learn new concepts over time. In this research, we conduct a comprehensive evaluation, showcasing the proficiency of Cobweb4V in learning visual concepts, requiring less data to achieve effective learning outcomes compared to traditional methods, maintaining stable performance over time, and achieving commendable asymptotic behavior, without catastrophic forgetting effects. These characteristics align with learning strategies in human cognition, positioning Cobweb4V as a promising alternative to neural network approaches.
    
[^3]: 观察学习：基于视频的机器人操作学习方法综述

    Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation

    [https://arxiv.org/abs/2402.07127](https://arxiv.org/abs/2402.07127)

    

    

    机器人学习操作技能受到多样化、无偏的数据集的稀缺性的影响。尽管策划的数据集可以帮助解决问题，但在泛化性和现实世界的转移方面仍然存在挑战。与此同时，“野外”视频数据集的大规模存在通过自监督技术推动了计算机视觉的进展。将这一点应用到机器人领域，最近的研究探索了通过被动观察来学习丰富的在线视频中的操作技能。这种基于视频的学习范式显示出了有希望的结果，它提供了可扩展的监督方法，同时降低了数据集的偏见。本综述回顾了视频特征表示学习技术、物体可行性理解、三维手部/身体建模和大规模机器人资源等基础知识，以及从不受控制的视频演示中获取机器人操作技能的新兴技术。我们讨论了仅从观察大规模人类视频中学习如何增强机器人的泛化性和样本效率。

    Robot learning of manipulation skills is hindered by the scarcity of diverse, unbiased datasets. While curated datasets can help, challenges remain in generalizability and real-world transfer. Meanwhile, large-scale "in-the-wild" video datasets have driven progress in computer vision through self-supervised techniques. Translating this to robotics, recent works have explored learning manipulation skills by passively watching abundant videos sourced online. Showing promising results, such video-based learning paradigms provide scalable supervision while reducing dataset bias. This survey reviews foundations such as video feature representation learning techniques, object affordance understanding, 3D hand/body modeling, and large-scale robot resources, as well as emerging techniques for acquiring robot manipulation skills from uncontrolled video demonstrations. We discuss how learning only from observing large-scale human videos can enhance generalization and sample efficiency for roboti
    
[^4]: NN-Copula-CD：一种基于Copula的可解释神经网络用于异构遥感图像变化检测

    NN-Copula-CD: A Copula-Guided Interpretable Neural Network for Change Detection in Heterogeneous Remote Sensing Images. (arXiv:2303.17448v1 [cs.CV])

    [http://arxiv.org/abs/2303.17448](http://arxiv.org/abs/2303.17448)

    该论文提出了一种可解释的神经网络方法，结合Copula理论来解决异构遥感图像中的变化检测问题。

    

    异构遥感图像中的变化检测是一个实际而具有挑战性的问题。过去十年来，深度神经网络(DNN)的发展让异构变化检测问题受益匪浅。然而，数据驱动的DNN始终像黑匣子一样，缺乏可解释性，这限制了DNN在大多数实际变化检测应用中的可靠性和可控性。为了解决这些问题，我们提出了一种基于Copula的可解释神经网络异构变化检测方法(NN-Copula-CD)。在NN-Copula-CD中，Copula的数学特征被设计为损失函数，用于监督一个简单的全连接神经网络学习变量之间的相关性。

    Change detection (CD) in heterogeneous remote sensing images is a practical and challenging issue for real-life emergencies. In the past decade, the heterogeneous CD problem has significantly benefited from the development of deep neural networks (DNN). However, the data-driven DNNs always perform like a black box where the lack of interpretability limits the trustworthiness and controllability of DNNs in most practical CD applications. As a strong knowledge-driven tool to measure correlation between random variables, Copula theory has been introduced into CD, yet it suffers from non-robust CD performance without manual prior selection for Copula functions. To address the above issues, we propose a knowledge-data-driven heterogeneous CD method (NN-Copula-CD) based on the Copula-guided interpretable neural network. In our NN-Copula-CD, the mathematical characteristics of Copula are designed as the losses to supervise a simple fully connected neural network to learn the correlation betwe
    
[^5]: LLaMA-Adapter: 零初始化注意力下的语言模型精细调整的高效方法

    LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. (arXiv:2303.16199v1 [cs.CV])

    [http://arxiv.org/abs/2303.16199](http://arxiv.org/abs/2303.16199)

    本文提出了一种基于适应提示和零初始化注意力机制的轻量级语言模型调整方法，可高效微调LLaMA为指令跟随模型，具有比Alpaca更短的微调时间并具有近似的响应质量。

    

    本文提出了LLaMA-Adapter这一轻量级适应方法，用于将LLaMA高效地微调为一个指令跟随模型。利用52K个自我指导示范，LLaMA-Adapter仅在冻结的LLaMA 7B模型上引入了1.2M个可学习参数，并且在8个A100 GPU上仅耗时不到一个小时进行微调。具体而言，我们采用一组可学习的适应提示，并在较高的变压器层中将它们预置于输入文本令牌之前。然后，提出了一种零初始化注意力机制和零门控机制，该机制可以自适应地将新的指令提示注入LLaMA，并有效地保留了其预先训练的知识。通过高效训练，LLaMA-Adapter能够产生高质量的响应，与完全微调的7B参数的Alpaca相似。此外，我们的方法还可以简单地扩展到多模态输入，例如图像，用于图像相关的LLaMA，在ScienceQA上实现了更强的推理能力。我们在https://github.com/ZrrSkywalker/LLaMA-Adapt发布了我们的代码。

    We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the input text tokens at higher transformer layers. Then, a zero-init attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge. With efficient training, LLaMA-Adapter generates high-quality responses, comparable to Alpaca with fully fine-tuned 7B parameters. Furthermore, our approach can be simply extended to multi-modal input, e.g., images, for image-conditioned LLaMA, which achieves superior reasoning capacity on ScienceQA. We release our code at https://github.com/ZrrSkywalker/LLaMA-Adapt
    
[^6]: MM-SHAP：一种用于衡量视觉与语言模型和任务的多模态贡献的性能不可知度量

    MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks. (arXiv:2212.08158v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.08158](http://arxiv.org/abs/2212.08158)

    该论文提出了一种性能不可知的多模态得分方法MM-SHAP，可以可靠地量化多模态模型使用各自模态的比例，并应用于比较模型的平均多模态程度和衡量个体模型的贡献。实验结果表明单模态崩溃比以前认为的更为普遍，而MM-SHAP是分析VL模型多模态行为的有效工具。

    

    已知视觉和语言模型（VL）往往利用各自模态中的不稳定指标（例如由分布偏差引入）而不是专注于每个模态中的相关信息。如果单模态模型在VL任务上达到类似多模态模型的准确度，则表明所谓的单模态崩溃已经发生。然而，基于准确度的测试无法检测例如模型预测错误但模型使用了一个模态的相关信息。因此，我们提出了MM-SHAP，一种基于Shapley值的性能不可知多模态得分，可可靠地量化多模态模型使用各自模态的比例。我们将MM-SHAP应用于两种方式：（1）比较模型的平均多模态程度，（2）衡量不同任务和数据集的个体模型对各自模态的贡献。六个VL模型的实验（LXMERT、CLIP和四个ALBEF变体）表明单模态崩溃比我们以前认为的更为普遍。我们的结果还表明，MM-SHAP是揭示和分析VL模型多模态行为的有效工具。

    Vision and language models (VL) are known to exploit unrobust indicators in individual modalities (e.g., introduced by distributional biases) instead of focusing on relevant information in each modality. That a unimodal model achieves similar accuracy on a VL task to a multimodal one, indicates that so-called unimodal collapse occurred. However, accuracy-based tests fail to detect e.g., when the model prediction is wrong, while the model used relevant information from a modality. Instead, we propose MM-SHAP, a performance-agnostic multimodality score based on Shapley values that reliably quantifies in which proportions a multimodal model uses individual modalities. We apply MM-SHAP in two ways: (1) to compare models for their average degree of multimodality, and (2) to measure for individual models the contribution of individual modalities for different tasks and datasets. Experiments with six VL models -- LXMERT, CLIP and four ALBEF variants -- on four VL tasks highlight that unimodal
    

