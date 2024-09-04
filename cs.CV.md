# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey for Foundation Models in Autonomous Driving](https://rss.arxiv.org/abs/2402.01105) | 本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。 |
| [^2] | [Graph-Jigsaw Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection](https://arxiv.org/abs/2403.12172) | 提出了一种名为GiCiSAD的基于图拼图条件扩散模型，用于解决基于骨架的视频异常检测中的挑战。 |
| [^3] | [Tur[k]ingBench: A Challenge Benchmark for Web Agents](https://arxiv.org/abs/2403.11905) | Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。 |
| [^4] | [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models](https://arxiv.org/abs/2403.06764) | FastV是一种多功能即插即用方法，通过学习自适应注意力模式并在后续层中修剪视觉代币，极大地降低了计算成本，同时在各种图像和视频理解任务中不损失性能。 |
| [^5] | [MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms](https://arxiv.org/abs/2402.14154) | 该研究介绍了MM-Soc，一个旨在评估多模态大型语言模型（MLLMs）对社交媒体内容理解的综合基准，通过对十种大小变体的四个开源MLLMs进行详尽评估，发现了显著的性能差异。 |
| [^6] | [FRDiff : Feature Reuse for Universal Training-free Acceleration of Diffusion Models](https://arxiv.org/abs/2312.03517) | 引入了一种新的加速技术FRDiff，通过利用扩散模型的时间冗余性，重新使用具有高时间相似性的特征图，节省计算资源而不影响输出质量。 |
| [^7] | [Hidden Flaws Behind Expert-Level Accuracy of GPT-4 Vision in Medicine.](http://arxiv.org/abs/2401.08396) | GPT-4 Vision在医学领域中具有专家级准确度，但在图像理解方面存在缺陷。 |
| [^8] | [NutritionVerse: Empirical Study of Various Dietary Intake Estimation Approaches.](http://arxiv.org/abs/2309.07704) | 该论文介绍了NutritionVerse-Synth，这是一个拥有大规模合成食物图像数据集，其中包含了多种视角、模态和饮食注释，旨在解决目前饮食摄入估计方法的准确性和真实性问题。 |
| [^9] | [RefSAM: Efficiently Adapting Segmenting Anything Model for Referring Video Object Segmentation.](http://arxiv.org/abs/2307.00997) | 本文介绍了RefSAM模型，该模型通过在线方式从不同时间戳的多视图信息中加入SAM的潜力，探索其在指代视频对象分割（RVOS）中的应用。通过使用跨模态MLP和分层稠密注意模块，我们改进了SAM模型，实现了对不同形态的精确理解，并取得了令人印象深刻的性能表现。 |
| [^10] | [Distilling Knowledge for Short-to-Long Term Trajectory Prediction.](http://arxiv.org/abs/2305.08553) | 本文提出了一种新的方法Di-Long，用于解决长期轨迹预测中越来越不确定和不可预测的问题。该方法利用蒸馏短期轨迹模型预测器来指导训练过程中的长期轨迹预测学生网络。学生网络观察短序列并预测长轨迹，教师网络观察更长序列并预测剩余短目标轨迹。 |

# 详细

[^1]: 自动驾驶领域基础模型综述

    A Survey for Foundation Models in Autonomous Driving

    [https://rss.arxiv.org/abs/2402.01105](https://rss.arxiv.org/abs/2402.01105)

    本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。

    

    基于基础模型的出现，自然语言处理和计算机视觉领域发生了革命，为自动驾驶应用铺平了道路。本综述论文对40多篇研究论文进行了全面的回顾，展示了基础模型在提升自动驾驶中的作用。大型语言模型在自动驾驶的规划和仿真中发挥着重要作用，特别是通过其在推理、代码生成和翻译方面的能力。与此同时，视觉基础模型在关键任务中得到越来越广泛的应用，例如三维物体检测和跟踪，以及为仿真和测试创建逼真的驾驶场景。多模态基础模型可以整合多样的输入，展现出卓越的视觉理解和空间推理能力，对于端到端自动驾驶至关重要。本综述不仅提供了一个结构化的分类，根据模态和自动驾驶领域中的功能对基础模型进行分类，还深入研究了方法。

    The advent of foundation models has revolutionized the fields of natural language processing and computer vision, paving the way for their application in autonomous driving (AD). This survey presents a comprehensive review of more than 40 research papers, demonstrating the role of foundation models in enhancing AD. Large language models contribute to planning and simulation in AD, particularly through their proficiency in reasoning, code generation and translation. In parallel, vision foundation models are increasingly adapted for critical tasks such as 3D object detection and tracking, as well as creating realistic driving scenarios for simulation and testing. Multi-modal foundation models, integrating diverse inputs, exhibit exceptional visual understanding and spatial reasoning, crucial for end-to-end AD. This survey not only provides a structured taxonomy, categorizing foundation models based on their modalities and functionalities within the AD domain but also delves into the meth
    
[^2]: 基于图拼图条件扩散模型的基于骨架的视频异常检测

    Graph-Jigsaw Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection

    [https://arxiv.org/abs/2403.12172](https://arxiv.org/abs/2403.12172)

    提出了一种名为GiCiSAD的基于图拼图条件扩散模型，用于解决基于骨架的视频异常检测中的挑战。

    

    基于骨架的视频异常检测（SVAD）是计算机视觉中的一个关键任务。准确识别异常模式或事件使操作员能够及时检测可疑活动，从而增强安全性。然而，现有研究未能同时解决这些关键特性。本文引入了一种新颖、实用且轻量级的框架，即基于图拼图条件扩散模型的基于骨架的视频异常检测（GiCiSAD），以克服与SVAD相关的挑战。

    arXiv:2403.12172v1 Announce Type: cross  Abstract: Skeleton-based video anomaly detection (SVAD) is a crucial task in computer vision. Accurately identifying abnormal patterns or events enables operators to promptly detect suspicious activities, thereby enhancing safety. Achieving this demands a comprehensive understanding of human motions, both at body and region levels, while also accounting for the wide variations of performing a single action. However, existing studies fail to simultaneously address these crucial properties. This paper introduces a novel, practical and lightweight framework, namely Graph-Jigsaw Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection (GiCiSAD) to overcome the challenges associated with SVAD. GiCiSAD consists of three novel modules: the Graph Attention-based Forecasting module to capture the spatio-temporal dependencies inherent in the data, the Graph-level Jigsaw Puzzle Maker module to distinguish subtle region-level discrepancies bet
    
[^3]: Tur[k]ingBench：用于网络代理的挑战基准测试

    Tur[k]ingBench: A Challenge Benchmark for Web Agents

    [https://arxiv.org/abs/2403.11905](https://arxiv.org/abs/2403.11905)

    Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。

    

    最近的聊天机器人展示了在原始文本形式下理解和交流的令人印象深刻的能力。然而，世界上不仅仅是原始文本。例如，人们在网页上花费大量时间，在这些网页上，文本与其他形式交织在一起，并以各种复杂互动的形式完成任务。最先进的多模型是否能够推广到这种复杂的领域呢？为了回答这个问题，我们介绍了TurkingBench，一个由包含多模态背景的文本说明制定的任务基准。与现有的使用人工合成的网页的工作不同，这里我们使用最初设计用于各种注释目的的自然HTML页面。每个任务的HTML说明也被实例化为各种值（从众包任务获得）以形成任务的新实例。这个基准包含32.2K个实例。

    arXiv:2403.11905v1 Announce Type: new  Abstract: Recent chatbots have demonstrated impressive ability to understand and communicate in raw-text form. However, there is more to the world than raw text. For example, humans spend long hours of their time on web pages, where text is intertwined with other modalities and tasks are accomplished in the form of various complex interactions. Can state-of-the-art multi-modal models generalize to such complex domains?   To address this question, we introduce TurkingBench, a benchmark of tasks formulated as web pages containing textual instructions with multi-modal context. Unlike existing work which employs artificially synthesized web pages, here we use natural HTML pages that were originally designed for crowdsourcing workers for various annotation purposes. The HTML instructions of each task are also instantiated with various values (obtained from the crowdsourcing tasks) to form new instances of the task. This benchmark contains 32.2K instanc
    
[^4]: 一张图片在第二层之后价值1/2代币：针对大规模视觉语言模型的即插即用推理加速

    An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models

    [https://arxiv.org/abs/2403.06764](https://arxiv.org/abs/2403.06764)

    FastV是一种多功能即插即用方法，通过学习自适应注意力模式并在后续层中修剪视觉代币，极大地降低了计算成本，同时在各种图像和视频理解任务中不损失性能。

    

    在本研究中，我们发现大规模视觉语言模型（LVLMs）中的注意力计算存在低效现象，尤其是在知名模型如LLaVA-1.5、QwenVL-Chat和Video-LLaVA中。我们发现在流行的LVLMs的深层中，对视觉代币的注意力计算极其低效，暗示相较于处理文本数据，需要更稀疏的方法。为此，我们引入了FastV，这是一种多功能即插即用方法，旨在通过学习早期层中的自适应注意力模式和在随后层中修剪视觉代币来优化计算效率。我们的评估表明FastV能够显著降低计算成本（例如，对于LLaVA-1.5-13B的FLOP减少了45%），而不会在广泛的图像和视频理解任务中牺牲性能。FastV的计算效率和性能权衡是高度可定制的，并且是帕累托有效的。

    arXiv:2403.06764v1 Announce Type: cross  Abstract: In this study, we identify the inefficient attention phenomena in Large Vision-Language Models (LVLMs), notably within prominent models like LLaVA-1.5, QwenVL-Chat and Video-LLaVA. We find out that the attention computation over visual tokens is of extreme inefficiency in the deep layers of popular LVLMs, suggesting a need for a sparser approach compared to textual data handling. To this end, we introduce FastV, a versatile plug-and-play method designed to optimize computational efficiency by learning adaptive attention patterns in early layers and pruning visual tokens in subsequent ones. Our evaluations demonstrate FastV's ability to dramatically reduce computational costs (e.g., a 45 reduction in FLOPs for LLaVA-1.5-13B) without sacrificing performance in a wide range of image and video understanding tasks. The computational efficiency and performance trade-off of FastV are highly customizable and pareto-efficient. It can compress t
    
[^5]: 在社交媒体平台上对多模态大型语言模型进行基准测试

    MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms

    [https://arxiv.org/abs/2402.14154](https://arxiv.org/abs/2402.14154)

    该研究介绍了MM-Soc，一个旨在评估多模态大型语言模型（MLLMs）对社交媒体内容理解的综合基准，通过对十种大小变体的四个开源MLLMs进行详尽评估，发现了显著的性能差异。

    

    社交媒体平台是多模态信息交流的中心，包括文本、图片和视频，这使得机器难以理解在线空间中交互所关联的信息或情绪。多模态大型语言模型（MLLMs）已经成为解决这些挑战的一个有前途的解决方案，但是它们在准确解释人类情绪和诸如虚假信息等复杂内容方面存在困难。本文介绍了MM-Soc，一个旨在评估MLLMs对多模态社交媒体内容理解的综合基准。MM-Soc整合了著名的多模态数据集，并融入了一个新颖的大规模YouTube标记数据集，旨在针对从虚假信息检测、仇恨言论检测到社交上下文生成等一系列任务。通过对四个开源MLLMs的十种不同规模变体进行详尽评估，我们发现了显著的性能差异，凸显出了对性能平衡的需求。

    arXiv:2402.14154v1 Announce Type: new  Abstract: Social media platforms are hubs for multimodal information exchange, encompassing text, images, and videos, making it challenging for machines to comprehend the information or emotions associated with interactions in online spaces. Multimodal Large Language Models (MLLMs) have emerged as a promising solution to address these challenges, yet struggle with accurately interpreting human emotions and complex contents like misinformation. This paper introduces MM-Soc, a comprehensive benchmark designed to evaluate MLLMs' understanding of multimodal social media content. MM-Soc compiles prominent multimodal datasets and incorporates a novel large-scale YouTube tagging dataset, targeting a range of tasks from misinformation detection, hate speech detection, and social context generation. Through our exhaustive evaluation on ten size-variants of four open-source MLLMs, we have identified significant performance disparities, highlighting the need
    
[^6]: FRDiff：特征重用用于无训练加速扩散模型

    FRDiff : Feature Reuse for Universal Training-free Acceleration of Diffusion Models

    [https://arxiv.org/abs/2312.03517](https://arxiv.org/abs/2312.03517)

    引入了一种新的加速技术FRDiff，通过利用扩散模型的时间冗余性，重新使用具有高时间相似性的特征图，节省计算资源而不影响输出质量。

    

    扩散模型的较大计算成本，特别是由于高质量图像生成所必需的重复去噪步骤而产生的，这是阻碍它们得到广泛采用的主要障碍。我们引入一种高级加速技术，利用扩散模型固有的时间冗余性来重新使用具有高时间相似性的特征图，从而节省计算资源而不影响输出质量。

    arXiv:2312.03517v2 Announce Type: replace-cross  Abstract: The substantial computational costs of diffusion models, especially due to the repeated denoising steps necessary for high-quality image generation, present a major obstacle to their widespread adoption. While several studies have attempted to address this issue by reducing the number of score function evaluations (NFE) using advanced ODE solvers without fine-tuning, the decreased number of denoising iterations misses the opportunity to update fine details, resulting in noticeable quality degradation. In our work, we introduce an advanced acceleration technique that leverages the temporal redundancy inherent in diffusion models. Reusing feature maps with high temporal similarity opens up a new opportunity to save computation resources without compromising output quality. To realize the practical benefits of this intuition, we conduct an extensive analysis and propose a novel method, FRDiff. FRDiff is designed to harness the adv
    
[^7]: GPT-4 Vision在医学领域中专家级准确度背后的隐藏缺陷

    Hidden Flaws Behind Expert-Level Accuracy of GPT-4 Vision in Medicine. (arXiv:2401.08396v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2401.08396](http://arxiv.org/abs/2401.08396)

    GPT-4 Vision在医学领域中具有专家级准确度，但在图像理解方面存在缺陷。

    

    最近的研究表明，具有Vision功能的GPT-4在医学挑战任务中表现优于人类医生。然而，这些评估主要关注多项选择题的准确度。本研究通过对GPT-4V在解决新英格兰医学杂志图像挑战中的图像理解、医学知识回忆和逐步多模态推理的原理进行全面分析，扩展了当前的研究范围。评估结果证实，GPT-4V在多项选择准确度上优于人类医生（88.0% vs. 77.0%，p=0.034）。GPT-4V在医生回答错误的情况下，也能表现出超过80%的准确度。然而，我们发现，GPT-4V在最终做出正确选择的情况下，经常提供有缺陷的推理（27.3%），其中最突出的是图像理解（21.6%）。

    Recent studies indicate that Generative Pre-trained Transformer 4 with Vision (GPT-4V) outperforms human physicians in medical challenge tasks. However, these evaluations primarily focused on the accuracy of multi-choice questions alone. Our study extends the current scope by conducting a comprehensive analysis of GPT-4V's rationales of image comprehension, recall of medical knowledge, and step-by-step multimodal reasoning when solving New England Journal of Medicine (NEJM) Image Challenges - an imaging quiz designed to test the knowledge and diagnostic capabilities of medical professionals. Evaluation results confirmed that GPT-4V outperforms human physicians regarding multi-choice accuracy (88.0% vs. 77.0%, p=0.034). GPT-4V also performs well in cases where physicians incorrectly answer, with over 80% accuracy. However, we discovered that GPT-4V frequently presents flawed rationales in cases where it makes the correct final choices (27.3%), most prominent in image comprehension (21.6
    
[^8]: NutritionVerse: 各种饮食摄入估计方法的实证研究

    NutritionVerse: Empirical Study of Various Dietary Intake Estimation Approaches. (arXiv:2309.07704v1 [cs.CV])

    [http://arxiv.org/abs/2309.07704](http://arxiv.org/abs/2309.07704)

    该论文介绍了NutritionVerse-Synth，这是一个拥有大规模合成食物图像数据集，其中包含了多种视角、模态和饮食注释，旨在解决目前饮食摄入估计方法的准确性和真实性问题。

    

    准确的饮食摄入估计对于支持健康饮食的政策和程序至关重要，因为营养不良与生活质量下降直接相关。然而，诸如食物日记之类的自我报告方法存在显著偏差。其他传统的饮食评估技术和新兴的替代方法，如移动应用程序，耗时长，并且可能需要受过训练的人员。最近的研究集中于使用计算机视觉和机器学习来从食物图像中自动估计饮食摄入量，但缺乏具有多样视角、模态和食物注释的综合数据集限制了这种方法的准确性和真实性。为了解决这个局限性，我们引入了NutritionVerse-Synth，这是第一个拥有84,984个逼真的合成2D食物图像及相关饮食信息和多模态标注的大规模数据集（包括深度图像、实例掩膜和语义掩膜）。

    Accurate dietary intake estimation is critical for informing policies and programs to support healthy eating, as malnutrition has been directly linked to decreased quality of life. However self-reporting methods such as food diaries suffer from substantial bias. Other conventional dietary assessment techniques and emerging alternative approaches such as mobile applications incur high time costs and may necessitate trained personnel. Recent work has focused on using computer vision and machine learning to automatically estimate dietary intake from food images, but the lack of comprehensive datasets with diverse viewpoints, modalities and food annotations hinders the accuracy and realism of such methods. To address this limitation, we introduce NutritionVerse-Synth, the first large-scale dataset of 84,984 photorealistic synthetic 2D food images with associated dietary information and multimodal annotations (including depth images, instance masks, and semantic masks). Additionally, we col
    
[^9]: RefSAM：高效适应任何模型的指代视频对象分割

    RefSAM: Efficiently Adapting Segmenting Anything Model for Referring Video Object Segmentation. (arXiv:2307.00997v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.00997](http://arxiv.org/abs/2307.00997)

    本文介绍了RefSAM模型，该模型通过在线方式从不同时间戳的多视图信息中加入SAM的潜力，探索其在指代视频对象分割（RVOS）中的应用。通过使用跨模态MLP和分层稠密注意模块，我们改进了SAM模型，实现了对不同形态的精确理解，并取得了令人印象深刻的性能表现。

    

    Segment Anything Model (SAM)因其在图像分割中出色的性能而引起了广泛关注。然而，在指代视频对象分割（RVOS）方面，由于需要精确的用户交互提示以及对语言和视觉等不同形态的有限理解能力，SAM缺乏熟练度。本文提出了RefSAM模型，通过在线方式从不同时间戳的多视图信息中加入SAM的潜力，探索其在RVOS中的应用。我们的方法对原始SAM模型进行了适应，通过使用轻量级的跨模态MLP将指代表达的文本嵌入投影为稀疏和密集嵌入，作为用户交互提示，以增强跨模态学习。此外，我们还引入了分层稠密注意模块，以将分层视觉语义信息与稀疏嵌入融合，以获得细粒度的密集嵌入。

    The Segment Anything Model (SAM) has gained significant attention for its impressive performance in image segmentation. However, it lacks proficiency in referring video object segmentation (RVOS) due to the need for precise user-interactive prompts and a limited understanding of different modalities, such as language and vision. This paper presents the RefSAM model, which explores the potential of SAM for RVOS by incorporating multi-view information from diverse modalities and successive frames at different timestamps in an online manner. Our proposed approach adapts the original SAM model to enhance cross-modality learning by employing a lightweight Cross-Modal MLP that projects the text embedding of the referring expression into sparse and dense embeddings, serving as user-interactive prompts. Additionally, we have introduced the hierarchical dense attention module to fuse hierarchical visual semantic information with sparse embeddings in order to obtain fine-grained dense embeddings
    
[^10]: 将知识蒸馏用于短期到长期轨迹预测

    Distilling Knowledge for Short-to-Long Term Trajectory Prediction. (arXiv:2305.08553v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.08553](http://arxiv.org/abs/2305.08553)

    本文提出了一种新的方法Di-Long，用于解决长期轨迹预测中越来越不确定和不可预测的问题。该方法利用蒸馏短期轨迹模型预测器来指导训练过程中的长期轨迹预测学生网络。学生网络观察短序列并预测长轨迹，教师网络观察更长序列并预测剩余短目标轨迹。

    

    长期轨迹预测是计算机视觉、机器学习和机器人领域中一个重要且具有挑战性的问题。其中一个基本困难在于随着时间范围的增长，轨迹的演变变得越来越不确定和不可预测，从而增加了问题的复杂性。为了克服这个问题，在本文中，我们提出了Di-Long，一种新的方法，它利用蒸馏短期轨迹模型预测器来指导训练过程中的长期轨迹预测学生网络。给定一个包含学生网络允许的观测序列和补充目标序列的总序列长度，我们让学生和教师对同一个完整轨迹定义两个不同但相关的任务：学生观察一个短序列并预测一个长轨迹，而教师观察一个更长的序列并预测剩下的短目标轨迹。

    Long-term trajectory forecasting is an important and challenging problem in the fields of computer vision, machine learning, and robotics. One fundamental difficulty stands in the evolution of the trajectory that becomes more and more uncertain and unpredictable as the time horizon grows, subsequently increasing the complexity of the problem. To overcome this issue, in this paper, we propose Di-Long, a new method that employs the distillation of a short-term trajectory model forecaster that guides a student network for long-term trajectory prediction during the training process. Given a total sequence length that comprehends the allowed observation for the student network and the complementary target sequence, we let the student and the teacher solve two different related tasks defined over the same full trajectory: the student observes a short sequence and predicts a long trajectory, whereas the teacher observes a longer sequence and predicts the remaining short target trajectory. The
    

