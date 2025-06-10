# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^2] | [Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild](https://arxiv.org/abs/2403.14539) | 提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。 |
| [^3] | [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://arxiv.org/abs/2402.13217) | VideoPrism是一个通用的视频编码器，通过全局-局部语义视频嵌入的蒸馏和标记混洗方案，在多个视频理解任务上取得了最新技术水平的表现。 |
| [^4] | [Zero-shot Object-Level OOD Detection with Context-Aware Inpainting](https://arxiv.org/abs/2402.03292) | 本论文提出了一种用上下文感知修复的零样本物体级OOD检测方法RONIN。通过将检测到的对象进行修复替换，并使用预测的ID标签来条件化修复过程，使得重构的对象在OOD情况下与原始对象相差较远，从而有效区分ID和OOD样本。实验证明RONIN在多个数据集上取得了具有竞争力的结果。 |
| [^5] | [Sequential Experimental Design for X-Ray CT Using Deep Reinforcement Learning.](http://arxiv.org/abs/2307.06343) | 本论文提出了一种使用深度强化学习的顺序实验设计方法，该方法可以在X射线CT中减少扫描角度的数量同时保持重建质量，从而适用于在线质量控制。 |

# 详细

[^1]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^2]: Object-Centric Domain Randomization用于野外3D形状重建

    Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild

    [https://arxiv.org/abs/2403.14539](https://arxiv.org/abs/2403.14539)

    提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。

    

    单视图3D形状在野外的重建面临的最大挑战之一是来自真实环境中的<3D形状，2D图像>-配对数据的稀缺性。受域随机化引人注目的成就的启发，我们提出了ObjectDR，通过对对象外观和背景的视觉变化进行随机仿真，合成这种配对数据。我们的数据合成框架利用条件生成模型（例如ControlNet）生成符合空间条件（例如2.5D草图）的图像，这些条件可以通过从对象集合（例如Objaverse-XL）的渲染过程获得3D形状。为了模拟多样化的变化同时保留嵌入空间条件中的对象轮廓，我们还引入了一个利用初始对象指导的解耦框架。

    arXiv:2403.14539v1 Announce Type: cross  Abstract: One of the biggest challenges in single-view 3D shape reconstruction in the wild is the scarcity of <3D shape, 2D image>-paired data from real-world environments. Inspired by remarkable achievements via domain randomization, we propose ObjectDR which synthesizes such paired data via a random simulation of visual variations in object appearances and backgrounds. Our data synthesis framework exploits a conditional generative model (e.g., ControlNet) to generate images conforming to spatial conditions such as 2.5D sketches, which are obtainable through a rendering process of 3D shapes from object collections (e.g., Objaverse-XL). To simulate diverse variations while preserving object silhouettes embedded in spatial conditions, we also introduce a disentangled framework which leverages an initial object guidance. After synthesizing a wide range of data, we pre-train a model on them so that it learns to capture a domain-invariant geometry p
    
[^3]: VideoPrism: 用于视频理解的基础视觉编码器

    VideoPrism: A Foundational Visual Encoder for Video Understanding

    [https://arxiv.org/abs/2402.13217](https://arxiv.org/abs/2402.13217)

    VideoPrism是一个通用的视频编码器，通过全局-局部语义视频嵌入的蒸馏和标记混洗方案，在多个视频理解任务上取得了最新技术水平的表现。

    

    我们引入了VideoPrism，一个通用的视频编码器，使用单个冻结模型处理多样的视频理解任务。我们在包含3600万高质量视频标题对和58.2亿个带有嘈杂平行文本（如ASR转录）的视频剪辑的异构语料库上对VideoPrism进行预训练。预训练方法通过全局-局部语义视频嵌入的蒸馏和一个标记混洗方案改进了掩码自编码，使VideoPrism能够主要专注于视频模态同时利用与视频相关联的宝贵文本。我们在四个广泛的视频理解任务组上进行了对VideoPrism的广泛测试，从网络视频问答到科学CV， 在33个视频理解基准测试中的30个上实现了最新技术水平的性能。

    arXiv:2402.13217v1 Announce Type: cross  Abstract: We introduce VideoPrism, a general-purpose video encoder that tackles diverse video understanding tasks with a single frozen model. We pretrain VideoPrism on a heterogeneous corpus containing 36M high-quality video-caption pairs and 582M video clips with noisy parallel text (e.g., ASR transcripts). The pretraining approach improves upon masked autoencoding by global-local distillation of semantic video embeddings and a token shuffling scheme, enabling VideoPrism to focus primarily on the video modality while leveraging the invaluable text associated with videos. We extensively test VideoPrism on four broad groups of video understanding tasks, from web video question answering to CV for science, achieving state-of-the-art performance on 30 out of 33 video understanding benchmarks.
    
[^4]: 用上下文感知修复的零样本物体级OOD检测

    Zero-shot Object-Level OOD Detection with Context-Aware Inpainting

    [https://arxiv.org/abs/2402.03292](https://arxiv.org/abs/2402.03292)

    本论文提出了一种用上下文感知修复的零样本物体级OOD检测方法RONIN。通过将检测到的对象进行修复替换，并使用预测的ID标签来条件化修复过程，使得重构的对象在OOD情况下与原始对象相差较远，从而有效区分ID和OOD样本。实验证明RONIN在多个数据集上取得了具有竞争力的结果。

    

    机器学习算法越来越多地作为黑盒云服务或预训练模型提供，无法访问它们的训练数据。这就引发了零样本离群数据（OOD）检测的问题。具体而言，我们的目标是检测不属于分类器标签集但被错误地归类为入域（ID）对象的OOD对象。我们的方法RONIN使用现成的扩散模型来用修复替换掉检测到的对象。RONIN使用预测的ID标签来条件化修复过程，使输入对象接近入域域。结果是，重构的对象在ID情况下非常接近原始对象，在OOD情况下则相差较远，使得RONIN能够有效区分ID和OOD样本。通过大量实验证明，RONIN在零样本和非零样本设置下，相对于先前方法，在多个数据集上取得了具有竞争力的结果。

    Machine learning algorithms are increasingly provided as black-box cloud services or pre-trained models, without access to their training data. This motivates the problem of zero-shot out-of-distribution (OOD) detection. Concretely, we aim to detect OOD objects that do not belong to the classifier's label set but are erroneously classified as in-distribution (ID) objects. Our approach, RONIN, uses an off-the-shelf diffusion model to replace detected objects with inpainting. RONIN conditions the inpainting process with the predicted ID label, drawing the input object closer to the in-distribution domain. As a result, the reconstructed object is very close to the original in the ID cases and far in the OOD cases, allowing RONIN to effectively distinguish ID and OOD samples. Throughout extensive experiments, we demonstrate that RONIN achieves competitive results compared to previous approaches across several datasets, both in zero-shot and non-zero-shot settings.
    
[^5]: 使用深度强化学习的X射线CT顺序实验设计

    Sequential Experimental Design for X-Ray CT Using Deep Reinforcement Learning. (arXiv:2307.06343v1 [eess.IV])

    [http://arxiv.org/abs/2307.06343](http://arxiv.org/abs/2307.06343)

    本论文提出了一种使用深度强化学习的顺序实验设计方法，该方法可以在X射线CT中减少扫描角度的数量同时保持重建质量，从而适用于在线质量控制。

    

    在X射线计算机断层扫描（CT）中，需从多个角度获取投影，并用于三维重建。为了使CT适用于在线质量控制，需要减少角度数目同时保持重建质量。稀疏角度断层扫描是从有限数据获取三维重建的常用方法。为了优化其性能，可以按序适应扫描角度，选择每个扫描对象最有信息量的角度。数学上，这对应于解决一个最优实验设计（OED）问题。OED问题是高维、非凸、双层优化问题，无法在线解决，即无法在扫描过程中解决。为了解决这些挑战，我们将OED问题在贝叶斯框架中建模为一个部分可观测马尔可夫决策过程，并通过深度强化学习来求解。该方法通过大量离线训练学习高效的非贪婪策略来解决给定类别的OED问题。

    In X-ray Computed Tomography (CT), projections from many angles are acquired and used for 3D reconstruction. To make CT suitable for in-line quality control, reducing the number of angles while maintaining reconstruction quality is necessary. Sparse-angle tomography is a popular approach for obtaining 3D reconstructions from limited data. To optimize its performance, one can adapt scan angles sequentially to select the most informative angles for each scanned object. Mathematically, this corresponds to solving and optimal experimental design (OED) problem. OED problems are high-dimensional, non-convex, bi-level optimization problems that cannot be solved online, i.e., during the scan. To address these challenges, we pose the OED problem as a partially observable Markov decision process in a Bayesian framework, and solve it through deep reinforcement learning. The approach learns efficient non-greedy policies to solve a given class of OED problems through extensive offline training rath
    

