# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Forward Learning for Gradient-based Black-box Saliency Map Generation](https://arxiv.org/abs/2403.15603) | 提出了一种新颖的统一框架，在黑盒设置中估计梯度并生成显著图解释模型决策，通过Likelihood Ratio方法估计输出到输入的梯度，并应用分块计算技术提高估计准确性，实验证实有效性和可扩展性。 |
| [^2] | [Accelerating Diffusion Sampling with Optimized Time Steps](https://arxiv.org/abs/2402.17376) | 提出了一个通用框架用于设计优化问题，旨在通过寻找更合适的时间步长加速扩散采样。 |
| [^3] | [Knowledge Transfer from Vision Foundation Models for Efficient Training of Small Task-specific Models](https://arxiv.org/abs/2311.18237) | 本文提出了一个简单的任务导向的知识迁移方法，用于高效训练小型任务特定模型。实验结果表明，该方法在多个目标任务上表现出了更好的性能，并且还展示了高达9倍的性能提升。 |
| [^4] | [LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding](https://arxiv.org/abs/2311.15876) | RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。 |
| [^5] | [Interpretable Semiotics Networks Representing Awareness](https://arxiv.org/abs/2310.05212) | 这个研究描述了一个计算模型，通过追踪和模拟物体感知以及其在交流中所传达的表示来模拟人类的意识。相比于大多数无法解释的神经网络，该模型具有解释性，并可以通过构建新网络来定义物体感知。 |
| [^6] | [SINCERE: Supervised Information Noise-Contrastive Estimation REvisited](https://arxiv.org/abs/2309.14277) | SINCERE提出了一个理论上合理的监督扩展，避免了同一类别的图像相互排斥，通过更好地分离不同类别的嵌入，在保持竞争性分类准确性的同时实现了更好的效果。 |
| [^7] | [AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents.](http://arxiv.org/abs/2401.12963) | AutoRT是一个利用现有的基础模型来扩展机器人在未知场景中的部署的系统，通过利用视觉-语言模型和大型语言模型，提出多样化和新颖的指令，并有效地推理自主权和安全性的权衡。 |
| [^8] | [Enhancing Deep Neural Network Training Efficiency and Performance through Linear Prediction.](http://arxiv.org/abs/2310.10958) | 本文提出了一种通过线性预测来提高深度神经网络训练效率和性能的方法。实验结果表明，在相同的训练条件和时期下，通过采用该方法可以提高模型的性能。 |
| [^9] | [Towards Robust Cardiac Segmentation using Graph Convolutional Networks.](http://arxiv.org/abs/2310.01210) | 这项研究使用了图卷积网络来实现心脏分割，通过预测轮廓点而不是标记每个像素，消除了心脏分割中的解剖学错误。同时还对图卷积网络进行了消融研究，并评估了临床测量指标的性能。 |

# 详细

[^1]: 基于前向学习的基于梯度的黑盒显著图生成

    Forward Learning for Gradient-based Black-box Saliency Map Generation

    [https://arxiv.org/abs/2403.15603](https://arxiv.org/abs/2403.15603)

    提出了一种新颖的统一框架，在黑盒设置中估计梯度并生成显著图解释模型决策，通过Likelihood Ratio方法估计输出到输入的梯度，并应用分块计算技术提高估计准确性，实验证实有效性和可扩展性。

    

    梯度-based显著图被广泛用于解释深度神经网络决策。然而，随着模型变得更深和更黑盒，如在闭源API（如ChatGPT）中，计算梯度变得具有挑战性，阻碍传统解释方法。在这项工作中，我们引入了一个新颖的统一框架，用于在黑盒设置中估计梯度并生成显著图来解释模型决策。我们采用似然比方法来估计输出到输入的梯度，并将其用于显著图生成。此外，我们提出了分块计算技术来增强估计准确性。在黑盒设置中进行的大量实验证实了我们方法的有效性，展示了准确的梯度估计和生成显著图的解释性。此外，我们通过应用它来解释GPT-Vision展示了我们方法的可扩展性，揭示了梯度相关性的持续影响。

    arXiv:2403.15603v1 Announce Type: cross  Abstract: Gradient-based saliency maps are widely used to explain deep neural network decisions. However, as models become deeper and more black-box, such as in closed-source APIs like ChatGPT, computing gradients become challenging, hindering conventional explanation methods. In this work, we introduce a novel unified framework for estimating gradients in black-box settings and generating saliency maps to interpret model decisions. We employ the likelihood ratio method to estimate output-to-input gradients and utilize them for saliency map generation. Additionally, we propose blockwise computation techniques to enhance estimation accuracy. Extensive experiments in black-box settings validate the effectiveness of our method, demonstrating accurate gradient estimation and explainability of generated saliency maps. Furthermore, we showcase the scalability of our approach by applying it to explain GPT-Vision, revealing the continued relevance of gr
    
[^2]: 优化时间步长加速扩散采样

    Accelerating Diffusion Sampling with Optimized Time Steps

    [https://arxiv.org/abs/2402.17376](https://arxiv.org/abs/2402.17376)

    提出了一个通用框架用于设计优化问题，旨在通过寻找更合适的时间步长加速扩散采样。

    

    扩散概率模型（DPMs）在高分辨率图像合成中表现出色，但由于通常需要大量采样步骤，其采样效率仍有待提高。近期高阶数值ODE求解器在DPMs中的应用使得用更少的采样步骤生成高质量图像成为可能。尽管这是一项重大进展，大多数采样方法仍然采用均匀时间步长，而在采样步骤较少时并不是最佳选择。为解决这一问题，我们提出了一个通用框架，用于设计一个优化问题，该优化问题旨在为DPMs的特定数值ODE求解器寻找更合适的时间步长。此优化问题旨在最小化地实现地真实解与与数值求解器对应的近似解之间的距离。它可以通过受限信赖域方法进行高效求解，时间少于

    arXiv:2402.17376v1 Announce Type: cross  Abstract: Diffusion probabilistic models (DPMs) have shown remarkable performance in high-resolution image synthesis, but their sampling efficiency is still to be desired due to the typically large number of sampling steps. Recent advancements in high-order numerical ODE solvers for DPMs have enabled the generation of high-quality images with much fewer sampling steps. While this is a significant development, most sampling methods still employ uniform time steps, which is not optimal when using a small number of steps. To address this issue, we propose a general framework for designing an optimization problem that seeks more appropriate time steps for a specific numerical ODE solver for DPMs. This optimization problem aims to minimize the distance between the ground-truth solution to the ODE and an approximate solution corresponding to the numerical solver. It can be efficiently solved using the constrained trust region method, taking less than 
    
[^3]: 从视觉基础模型中进行知识迁移用于高效训练小型任务特定模型

    Knowledge Transfer from Vision Foundation Models for Efficient Training of Small Task-specific Models

    [https://arxiv.org/abs/2311.18237](https://arxiv.org/abs/2311.18237)

    本文提出了一个简单的任务导向的知识迁移方法，用于高效训练小型任务特定模型。实验结果表明，该方法在多个目标任务上表现出了更好的性能，并且还展示了高达9倍的性能提升。

    

    在许多下游任务中，基于大规模数据集预训练的视觉基础模型在有限标记的目标数据上展现出了令人印象深刻的性能。然而，由于推理计算成本高，这些模型无法应用于许多实际应用。为了解决这个问题，我们提出了一个简单的任务导向的知识迁移方法，以高效解决如何利用大规模视觉基础模型的知识来训练小型任务特定模型的问题。我们在五个目标任务上的实验结果表明，该方法在超过Task-Agnostic VFM蒸馏、Web-Scale CLIP预训练、监督式ImageNet预训练和自监督DINO预训练29.8%、22.1%、13.7%和11.6%的方面表现出更好的性能。此外，所提出的方法还展现出了高达9倍的性能提升。

    arXiv:2311.18237v2 Announce Type: replace-cross  Abstract: Vision Foundation Models (VFMs) pretrained on massive datasets exhibit impressive performance on various downstream tasks, especially with limited labeled target data. However, due to their high inference compute cost, these models cannot be deployed for many real-world applications. Motivated by this, we ask the following important question, "How can we leverage the knowledge from a large VFM to train a small task-specific model for a new target task with limited labeled training data?", and propose a simple task-oriented knowledge transfer approach as a highly effective solution to this problem. Our experimental results on five target tasks show that the proposed approach outperforms task-agnostic VFM distillation, web-scale CLIP pretraining, supervised ImageNet pretraining, and self-supervised DINO pretraining by up to 11.6%, 22.1%, 13.7%, and 29.8%, respectively. Furthermore, the proposed approach also demonstrates up to 9x
    
[^4]: LMM辅助的一致性嵌入下乳腺癌治疗目标分割

    LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding

    [https://arxiv.org/abs/2311.15876](https://arxiv.org/abs/2311.15876)

    RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。

    

    人工智能的最新进展深刻影响了医学领域，为降低临床工作量提供了工具。然而，大多数人工智能模型受限于执行单模式任务，与医学专业人员所使用的综合方法形成鲜明对比。为解决这一问题，本文介绍了RO-LMM，一个专为放射肿瘤学领域设计的多功能大型多模型（LMM）。该模型涵盖了临床工作流中的一系列任务，擅长临床报告摘要、放疗治疗计划建议和计划引导的目标体积分割。为了执行连续的临床任务，我们进一步提出了一种新颖的一致性嵌入微调（CEFTune）技术，提升了LMM对嘈杂输入的鲁棒性，同时保持了处理干净输入的能力，并将该概念转化为LMM驱动的分割框架，即一致性嵌入S。

    arXiv:2311.15876v2 Announce Type: replace-cross  Abstract: Recent advancements in Artificial Intelligence (AI) have profoundly influenced medical fields, by providing tools to reduce clinical workloads. However, most AI models are constrained to execute unimodal tasks, in stark contrast to the comprehensive approaches utilized by medical professionals. To address this, here we present RO-LMM, a multi-purpose large multimodal model (LMM) tailored for the field of radiation oncology. This model covers series of tasks within clinical workflow, adept at clinical report summarization, radiation treatment plan suggestion, and plan-guided target volume segmentation. In particular, to perform consecutive clinical tasks, we further present a novel Consistency Embedding Fine-Tuning (CEFTune) technique, which boosts LMM's robustness to noisy inputs while preserving the capability of handling clean inputs, and transform this concept into LMM-driven segmentation framework as Consistency Embedding S
    
[^5]: 可解释的符号网络代表意识的知觉

    Interpretable Semiotics Networks Representing Awareness

    [https://arxiv.org/abs/2310.05212](https://arxiv.org/abs/2310.05212)

    这个研究描述了一个计算模型，通过追踪和模拟物体感知以及其在交流中所传达的表示来模拟人类的意识。相比于大多数无法解释的神经网络，该模型具有解释性，并可以通过构建新网络来定义物体感知。

    

    人类每天都感知物体，并通过各种渠道传达他们的感知。在这里，我们描述了一个计算模型，追踪和模拟物体的感知以及它们在交流中所传达的表示。我们描述了我们内部表示的两个关键组成部分（"观察到的"和"看到的"），并将它们与熟悉的计算机视觉概念（编码和解码）相关联。这些元素被合并在一起形成符号网络，模拟了物体感知和人类交流中的意识。如今，大多数神经网络都是不可解释的。另一方面，我们的模型克服了这个限制。实验证明了该模型的可见性。我们人的物体感知模型使我们能够通过网络定义物体感知。我们通过构建一个包括基准分类器和额外层的新网络来演示这一点。这个层产生了图像的感知。

    Humans perceive objects daily and communicate their perceptions using various channels. Here, we describe a computational model that tracks and simulates objects' perception and their representations as they are conveyed in communication.   We describe two key components of our internal representation ("observed" and "seen") and relate them to familiar computer vision notions (encoding and decoding). These elements are joined together to form semiotics networks, which simulate awareness in object perception and human communication.   Nowadays, most neural networks are uninterpretable. On the other hand, our model overcomes this limitation. The experiments demonstrates the visibility of the model.   Our model of object perception by a person allows us to define object perception by a network. We demonstrate this with an example of an image baseline classifier by constructing a new network that includes the baseline classifier and an additional layer. This layer produces the images "perc
    
[^6]: SINCERE: 监督信息噪声-对比估计再审

    SINCERE: Supervised Information Noise-Contrastive Estimation REvisited

    [https://arxiv.org/abs/2309.14277](https://arxiv.org/abs/2309.14277)

    SINCERE提出了一个理论上合理的监督扩展，避免了同一类别的图像相互排斥，通过更好地分离不同类别的嵌入，在保持竞争性分类准确性的同时实现了更好的效果。

    

    信息噪声对比估计（InfoNCE）损失函数由于其强大的实证结果和理论动机，为许多自监督深度学习方法提供了基础。先前的工作表明，监督对比（SupCon）损失可扩展InfoNCE以从可用类标签中学习。然而，在这项工作中，我们发现先前的SupCon损失公式存在疑问的理由，因为它可能会促使来自同一类别的某些图像在学习到的嵌入空间中相互排斥。我们提出了监督信息噪声-对比估计再审（SINCERE）损失，作为信息噪声对比估计的理论上合理的监督扩展，它永远不会导致来自同一类别的图像相互排斥。实验表明，SINCERE导致不同类别的嵌入更好地分离，同时对于监督和迁移学习提供具有竞争力的分类准确性。我们进一步展示了一个信息论上的下界

    arXiv:2309.14277v2 Announce Type: replace-cross  Abstract: The information noise-contrastive estimation (InfoNCE) loss function provides the basis of many self-supervised deep learning methods due to its strong empirical results and theoretic motivation. Previous work suggests a supervised contrastive (SupCon) loss to extend InfoNCE to learn from available class labels. However, in this work we find that the prior SupCon loss formulation has questionable justification because it can encourage some images from the same class to repel one another in the learned embedding space. We propose the Supervised InfoNCE REvisited (SINCERE) loss as a theoretically-justified supervised extension of InfoNCE that never causes images from the same class to repel one another. Experiments show that SINCERE leads to better separation of embeddings from different classes while delivering competitive classification accuracy for supervised and transfer learning. We further show an information-theoretic boun
    
[^7]: AutoRT：大规模编排机器人代理的具身基础模型

    AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents. (arXiv:2401.12963v1 [cs.RO])

    [http://arxiv.org/abs/2401.12963](http://arxiv.org/abs/2401.12963)

    AutoRT是一个利用现有的基础模型来扩展机器人在未知场景中的部署的系统，通过利用视觉-语言模型和大型语言模型，提出多样化和新颖的指令，并有效地推理自主权和安全性的权衡。

    

    拥有语言、视觉和行动等功能的具身基础模型已经彻底改变了利用互联网规模的数据来推理有用任务的能力。然而，训练具身基础模型的一个关键挑战是缺乏基于物理世界的数据。在本文中，我们提出了AutoRT，一个利用现有的基础模型来扩展完全未知场景中操作机器人的部署的系统，只需要最少的人工监督。AutoRT利用视觉-语言模型(VLMs)实现场景理解和基础绑定，并进一步利用大型语言模型(LLMs)提出多样化和新颖的指令，供一组机器人执行。通过利用基础模型的知识来指导数据收集，AutoRT能够有效地推理自主权和安全性的权衡，同时显著扩大机器人学习的数据收集。我们演示了AutoRT向20多个机器人提议指令。

    Foundation models that incorporate language, vision, and more recently actions have revolutionized the ability to harness internet scale data to reason about useful tasks. However, one of the key challenges of training embodied foundation models is the lack of data grounded in the physical world. In this paper, we propose AutoRT, a system that leverages existing foundation models to scale up the deployment of operational robots in completely unseen scenarios with minimal human supervision. AutoRT leverages vision-language models (VLMs) for scene understanding and grounding, and further uses large language models (LLMs) for proposing diverse and novel instructions to be performed by a fleet of robots. Guiding data collection by tapping into the knowledge of foundation models enables AutoRT to effectively reason about autonomy tradeoffs and safety while significantly scaling up data collection for robot learning. We demonstrate AutoRT proposing instructions to over 20 robots across multi
    
[^8]: 通过线性预测提高深度神经网络的训练效率和性能

    Enhancing Deep Neural Network Training Efficiency and Performance through Linear Prediction. (arXiv:2310.10958v1 [cs.LG])

    [http://arxiv.org/abs/2310.10958](http://arxiv.org/abs/2310.10958)

    本文提出了一种通过线性预测来提高深度神经网络训练效率和性能的方法。实验结果表明，在相同的训练条件和时期下，通过采用该方法可以提高模型的性能。

    

    深度神经网络（DNN）在计算机视觉和自然语言处理等领域取得了显著的成功。然而，训练一个有效的DNN模型仍然面临着挑战。本文旨在提出一种优化DNN训练效果的方法，旨在提高模型性能。首先，根据观察到的DNN参数在训练过程中遵循某种规律的观察，发现了参数预测可以提高模型训练效率和性能的潜力。其次，考虑到DNN模型参数的数量级、硬件限制和随机梯度下降（SGD）对噪声容忍度的特性，采用参数线性预测（PLP）方法来进行DNN参数预测。最后，在一些代表性的骨架上进行验证。实验结果表明，在相同的训练条件和时期下，与正常的训练方式相比，通过采用所提出的方法，能够提高模型的性能。

    Deep neural networks (DNN) have achieved remarkable success in various fields, including computer vision and natural language processing. However, training an effective DNN model still poses challenges. This paper aims to propose a method to optimize the training effectiveness of DNN, with the goal of improving model performance. Firstly, based on the observation that the DNN parameters change in certain laws during training process, the potential of parameter prediction for improving model training efficiency and performance is discovered. Secondly, considering the magnitude of DNN model parameters, hardware limitations and characteristics of Stochastic Gradient Descent (SGD) for noise tolerance, a Parameter Linear Prediction (PLP) method is exploit to perform DNN parameter prediction. Finally, validations are carried out on some representative backbones. Experiment results show that compare to the normal training ways, under the same training conditions and epochs, by employing propo
    
[^9]: 实现稳健的心脏分割：使用图卷积网络

    Towards Robust Cardiac Segmentation using Graph Convolutional Networks. (arXiv:2310.01210v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2310.01210](http://arxiv.org/abs/2310.01210)

    这项研究使用了图卷积网络来实现心脏分割，通过预测轮廓点而不是标记每个像素，消除了心脏分割中的解剖学错误。同时还对图卷积网络进行了消融研究，并评估了临床测量指标的性能。

    

    全自动化的心脏分割可以快速、可重复地从超声心动图检查中提取临床测量指标。U-Net结构是目前医学分割领域的深度学习架构，可以实时分割心脏结构，并且平均误差可与观测者间变异性相媲美。然而，该架构仍然会生成许多解离异常的结构。本研究使用图卷积神经网络的概念，预测出感兴趣结构的轮廓点，而不是对每个像素进行标记。我们提出了一个基于心脏解剖学的图结构，并证明这消除了公开可获得的CAMUS数据集上的多结构分割中的解剖学错误。此外，本研究还对图卷积网络进行了消融研究，并在临床HUNT4数据集上评估了临床测量指标。

    Fully automatic cardiac segmentation can be a fast and reproducible method to extract clinical measurements from an echocardiography examination. The U-Net architecture is the current state-of-the-art deep learning architecture for medical segmentation and can segment cardiac structures in real-time with average errors comparable to inter-observer variability. However, this architecture still generates large outliers that are often anatomically incorrect. This work uses the concept of graph convolutional neural networks that predict the contour points of the structures of interest instead of labeling each pixel. We propose a graph architecture that uses two convolutional rings based on cardiac anatomy and show that this eliminates anatomical incorrect multi-structure segmentations on the publicly available CAMUS dataset. Additionally, this work contributes with an ablation study on the graph convolutional architecture and an evaluation of clinical measurements on the clinical HUNT4 dat
    

