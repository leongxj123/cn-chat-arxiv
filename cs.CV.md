# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [X-Portrait: Expressive Portrait Animation with Hierarchical Motion Attention](https://arxiv.org/abs/2403.15931) | 这里是中文总结出的一句话要点: 该论文提出了X-Portrait，一种用于生成具有表现力和时间连贯性的肖像动画的条件扩散模型，利用控制信号实现了细粒度头部姿势和表情控制，以提高运动精度。 |
| [^2] | [Grounding Language Models for Visual Entity Recognition](https://arxiv.org/abs/2402.18695) | 通过AutoVER模型，我们提出了一种在视觉实体识别中应用自回归模型的方法，通过检索增强的约束生成，成功区分巨大标签空间中相似的实体，并在Oven-Wiki基准测试上取得显著进展。 |
| [^3] | [M3-VRD: Multimodal Multi-task Multi-teacher Visually-Rich Form Document Understanding](https://arxiv.org/abs/2402.17983) | 这个模型是一个多模态、多任务、多教师的联合细粒度知识蒸馏模型，通过微妙协作令牌和实体表示，处理复杂的表单文档，引入新的损失函数改进知识蒸馏过程，在处理视觉复杂表单文档的结构和内容上表现出色。 |
| [^4] | [Outlier detection by ensembling uncertainty with negative objectness](https://arxiv.org/abs/2402.15374) | 提出一种利用不确定性和负对象性集成的异常检测方法，通过直接预测K+1个logits并在密集预测结构中嵌入，可独立检测异常值。 |
| [^5] | [Model Composition for Multimodal Large Language Models](https://arxiv.org/abs/2402.12750) | 通过模型组合现有的多模态大型语言模型，提出了一种新范式，有效地保留了每个原始模型的模态理解能力，并引入了一种用于解决合并参数干扰和不匹配问题的方法。 |
| [^6] | [Diffusion MRI with Machine Learning](https://arxiv.org/abs/2402.00019) | 本文评估了机器学习在弥散磁共振成像中的应用，重点关注了微结构映射、纤维束描记、白质纤维束分析以及数据预处理和协调的方法。通过对现有方法的总结，提出了未来研究的主题。 |
| [^7] | [Learning to Visually Connect Actions and their Effects.](http://arxiv.org/abs/2401.10805) | 该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。 |
| [^8] | [Adaptive Self-training Framework for Fine-grained Scene Graph Generation.](http://arxiv.org/abs/2401.09786) | 本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。 |
| [^9] | [Viewpoint Textual Inversion: Unleashing Novel View Synthesis with Pretrained 2D Diffusion Models.](http://arxiv.org/abs/2309.07986) | 本研究展示了通过预训练的2D图像扩散模型，可以从仅有2D监督的情况下提取出3D结构信息，并利用该信息进行3D视觉任务。通过观点神经文本倒置（ViewNeTI）方法，我们可以控制生成图像中对象的3D视点，有效解决新颖视图合成问题，并在单视图情况下具有良好的语义细节和逼真度。 |
| [^10] | [Efficient OCR for Building a Diverse Digital History.](http://arxiv.org/abs/2304.02737) | 本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。 |
| [^11] | [Point-MA2E: Masked and Affine Transformed AutoEncoder for Self-supervised Point Cloud Learning.](http://arxiv.org/abs/2211.06841) | 本文介绍了一种点云学习的自监督方法Point-MA2E，通过同时采用掩膜和仿射变换策略，实现了从损坏点云到还原点云的重建，扩展了目前掩膜方法的不足。 |

# 详细

[^1]: X-Portrait: 具有分层动作注意力的表现性肖像动画

    X-Portrait: Expressive Portrait Animation with Hierarchical Motion Attention

    [https://arxiv.org/abs/2403.15931](https://arxiv.org/abs/2403.15931)

    这里是中文总结出的一句话要点: 该论文提出了X-Portrait，一种用于生成具有表现力和时间连贯性的肖像动画的条件扩散模型，利用控制信号实现了细粒度头部姿势和表情控制，以提高运动精度。

    

    我们提出了X-Portrait，这是一种创新的条件扩散模型，专门用于生成具有表现力和时间连贯性的肖像动画。具体而言，我们旨在基于单个肖像作为外观参考，并利用来自驱动视频的运动来为其添加动画，捕捉具有高度动态性和微妙面部表情以及广泛范围头部运动。在其核心部分，我们利用了预先训练的扩散模型的生成先验作为渲染骨架，同时在ControlNet框架内通过新颖的控制信号实现了细粒度头部姿势和表情控制。与传统的粗糙显式控制（如面部标志点）不同，我们的运动控制模块学会直接从原始驱动RGB输入中解读动态。通过有效增强对眼神等小尺度细微差异的运动关注的基于补丁的局部控制模块，进一步提高了运动精度。

    arXiv:2403.15931v1 Announce Type: cross  Abstract: We propose X-Portrait, an innovative conditional diffusion model tailored for generating expressive and temporally coherent portrait animation. Specifically, given a single portrait as appearance reference, we aim to animate it with motion derived from a driving video, capturing both highly dynamic and subtle facial expressions along with wide-range head movements. As its core, we leverage the generative prior of a pre-trained diffusion model as the rendering backbone, while achieve fine-grained head pose and expression control with novel controlling signals within the framework of ControlNet. In contrast to conventional coarse explicit controls such as facial landmarks, our motion control module is learned to interpret the dynamics directly from the original driving RGB inputs. The motion accuracy is further enhanced with a patch-based local control module that effectively enhance the motion attention to small-scale nuances like eyeba
    
[^2]: 将语言模型应用在视觉实体识别上

    Grounding Language Models for Visual Entity Recognition

    [https://arxiv.org/abs/2402.18695](https://arxiv.org/abs/2402.18695)

    通过AutoVER模型，我们提出了一种在视觉实体识别中应用自回归模型的方法，通过检索增强的约束生成，成功区分巨大标签空间中相似的实体，并在Oven-Wiki基准测试上取得显著进展。

    

    我们引入AutoVER，一种用于视觉实体识别的自回归模型。我们的模型通过使用检索增强的约束生成，扩展了自回归多模式大型语言模型。它在处理跨领域实体时减轻了低性能，在需要视觉推理的查询中表现出色。我们的方法通过在硬负对上进行对比训练，并在序列-序列目标中并行进行训练，学习在巨大的标签空间中区分相似的实体。在推断过程中，一系列检索的候选答案明确指导语言生成，通过消除无效的解码路径。所提出的方法在最近提出的Oven-Wiki基准测试的不同数据集拆分中实现了显著的改进。在已知实体拆分上的准确率从32.7%提高到61.5%。该方法还通过大幅度提升在未知和查询拆分上的性能，表现出卓越的表现。

    arXiv:2402.18695v1 Announce Type: cross  Abstract: We introduce AutoVER, an Autoregressive model for Visual Entity Recognition. Our model extends an autoregressive Multi-modal Large Language Model by employing retrieval augmented constrained generation. It mitigates low performance on out-of-domain entities while excelling in queries that require visually-situated reasoning. Our method learns to distinguish similar entities within a vast label space by contrastively training on hard negative pairs in parallel with a sequence-to-sequence objective without an external retriever. During inference, a list of retrieved candidate answers explicitly guides language generation by removing invalid decoding paths. The proposed method achieves significant improvements across different dataset splits in the recently proposed Oven-Wiki benchmark. Accuracy on the Entity seen split rises from 32.7% to 61.5%. It also demonstrates superior performance on the unseen and query splits by a substantial dou
    
[^3]: M3-VRD: 多模态多任务多教师视觉丰富表单文档理解

    M3-VRD: Multimodal Multi-task Multi-teacher Visually-Rich Form Document Understanding

    [https://arxiv.org/abs/2402.17983](https://arxiv.org/abs/2402.17983)

    这个模型是一个多模态、多任务、多教师的联合细粒度知识蒸馏模型，通过微妙协作令牌和实体表示，处理复杂的表单文档，引入新的损失函数改进知识蒸馏过程，在处理视觉复杂表单文档的结构和内容上表现出色。

    

    这篇论文提出了一个突破性的多模态、多任务、多教师联合细粒度知识蒸馏模型，用于视觉丰富的表单文档理解。该模型旨在通过促进令牌和实体表示之间的微妙相关性来利用细粒度和粗粒度级别的见解，解决表单文档固有的复杂性。此外，我们引入了新的跨细粒度和跨粗粒度损失函数，以进一步改进多教师知识蒸馏传递过程，呈现分布差距和对表单文档的统一理解。通过在公开可用的表单文档理解数据集上进行全面评估，我们提出的模型始终表现出色地优于现有基线，展示了其在处理复杂视觉表单文档的复杂结构和内容方面的功效。

    arXiv:2402.17983v1 Announce Type: new  Abstract: This paper presents a groundbreaking multimodal, multi-task, multi-teacher joint-grained knowledge distillation model for visually-rich form document understanding. The model is designed to leverage insights from both fine-grained and coarse-grained levels by facilitating a nuanced correlation between token and entity representations, addressing the complexities inherent in form documents. Additionally, we introduce new inter-grained and cross-grained loss functions to further refine diverse multi-teacher knowledge distillation transfer process, presenting distribution gaps and a harmonised understanding of form documents. Through a comprehensive evaluation across publicly available form document understanding datasets, our proposed model consistently outperforms existing baselines, showcasing its efficacy in handling the intricate structures and content of visually complex form documents.
    
[^4]: 利用不确定性和负对象性集成的异常检测

    Outlier detection by ensembling uncertainty with negative objectness

    [https://arxiv.org/abs/2402.15374](https://arxiv.org/abs/2402.15374)

    提出一种利用不确定性和负对象性集成的异常检测方法，通过直接预测K+1个logits并在密集预测结构中嵌入，可独立检测异常值。

    

    异常检测是监督式视觉识别中关键的功能。现有的大多数方法通过鼓励标准封闭集模型在负训练数据中产生低置信度预测来获得最佳结果。然而，这种方法混淆了预测不确定性和对负类别的识别。因此，我们重新考虑了直接预测K+1个logits，这些logits对应于K个基本真实类别和一个异常类别。这种设置允许我们制定一种新奇的异常得分，作为分布内不确定性和异常类别的后验的集合，我们称之为负对象性。现在，异常值可以通过高预测不确定性或与负数据相似之处独立检测。我们将我们的方法嵌入到一个密集预测结构中，该结构具有K+2个类别的掩码级别识别。训练过程鼓励新颖的K+2-th类别去学习

    arXiv:2402.15374v1 Announce Type: cross  Abstract: Outlier detection is an essential capability in safety-critical applications of supervised visual recognition. Most of the existing methods deliver best results by encouraging standard closed-set models to produce low-confidence predictions in negative training data. However, that approach conflates prediction uncertainty with recognition of the negative class. We therefore reconsider direct prediction of K+1 logits that correspond to K groundtruth classes and one outlier class. This setup allows us to formulate a novel anomaly score as an ensemble of in-distribution uncertainty and the posterior of the outlier class which we term negative objectness. Now outliers can be independently detected due to i) high prediction uncertainty or ii) similarity with negative data. We embed our method into a dense prediction architecture with mask-level recognition over K+2 classes. The training procedure encourages the novel K+2-th class to learn n
    
[^5]: 多模态大型语言模型的模型组合

    Model Composition for Multimodal Large Language Models

    [https://arxiv.org/abs/2402.12750](https://arxiv.org/abs/2402.12750)

    通过模型组合现有的多模态大型语言模型，提出了一种新范式，有效地保留了每个原始模型的模态理解能力，并引入了一种用于解决合并参数干扰和不匹配问题的方法。

    

    近期对多模态大型语言模型（MLLMs）的发展显示出了快速进展，朝着创建能够理解各种模态输入的多功能MLLMs的目标迈进。然而，现有方法通常依赖于与配对的多模态指令数据进行联合训练，这对资源要求高且难以扩展到新的模态。在本文中，我们提出了一种通过现有MLLMs的模型组合来创建一个新模型的新范式，该新模型保留了每个原始模型的模态理解能力。我们的基本实现NaiveMC通过重用模态编码器和合并LLM参数展示了这一范式的有效性。此外，我们引入了DAMC来解决在合并过程中的参数干扰和不匹配问题，从而提升了模型的性能。为促进该领域的研究，我们提出了MCUB，一个用于评估MLLMs理解能力的基准测试。

    arXiv:2402.12750v1 Announce Type: cross  Abstract: Recent developments in Multimodal Large Language Models (MLLMs) have shown rapid progress, moving towards the goal of creating versatile MLLMs that understand inputs from various modalities. However, existing methods typically rely on joint training with paired multimodal instruction data, which is resource-intensive and challenging to extend to new modalities. In this paper, we propose a new paradigm through the model composition of existing MLLMs to create a new model that retains the modal understanding capabilities of each original model. Our basic implementation, NaiveMC, demonstrates the effectiveness of this paradigm by reusing modality encoders and merging LLM parameters. Furthermore, we introduce DAMC to address parameter interference and mismatch issues during the merging process, thereby enhancing the model performance. To facilitate research in this area, we propose MCUB, a benchmark for assessing ability of MLLMs to unders
    
[^6]: 机器学习在弥散磁共振成像中的应用

    Diffusion MRI with Machine Learning

    [https://arxiv.org/abs/2402.00019](https://arxiv.org/abs/2402.00019)

    本文评估了机器学习在弥散磁共振成像中的应用，重点关注了微结构映射、纤维束描记、白质纤维束分析以及数据预处理和协调的方法。通过对现有方法的总结，提出了未来研究的主题。

    

    弥散加权磁共振成像（dMRI）具有非侵入性评估大脑微结构和结构连接的独特能力。然而，分析dMRI数据以提取临床和科学目的的有用信息具有挑战性。 dMRI测量通常受到强噪声和伪影的干扰，数据中通常存在高的会话间和扫描者间异质性，以及大脑结构的相当大的个体间变异，并且测量和感兴趣现象之间的关系可能非常复杂。近年来，机器学习方法在dMRI分析中的应用越来越多。本文旨在评估这些尝试，重点关注已经解决了微结构映射、纤维束描记、白质纤维束分析以及数据预处理和协调的方法。我们总结了现有方法的主要发现、优点和缺点，并提出了未来研究的主题。

    Diffusion-weighted magnetic resonance imaging (dMRI) offers unique capabilities such as noninvasive assessment of brain's micro-structure and structural connectivity. However, analyzing the dMRI data to extract useful information for clinical and scientific purposes is challenging. The dMRI measurements often suffer from strong noise and artifacts, there is usually high inter-session and inter-scanner heterogeneity in the data and considerable inter-subject variability in brain structure, and the relationship between measurements and the phenomena of interest can be highly complex. Recent years have witnessed increasing use of machine learning methods for dMRI analysis. This manuscript aims to assess these efforts, with a focus on methods that have addressed micro-structure mapping, tractography, white matter tract analysis, as well as data preprocessing and harmonization. We summarize the main findings, strengths, and weaknesses of the existing methods and suggest topics for future re
    
[^7]: 学习视觉连接动作和其效果

    Learning to Visually Connect Actions and their Effects. (arXiv:2401.10805v1 [cs.CV])

    [http://arxiv.org/abs/2401.10805](http://arxiv.org/abs/2401.10805)

    该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。

    

    在这项工作中，我们引入了视觉连接动作和其效果（CATE）的新概念，用于视频理解。CATE可以在任务规划和从示范中学习等领域中应用。我们提出了不同基于CATE的任务形式，如动作选择和动作指定，其中视频理解模型以语义和细粒度的方式连接动作和效果。我们观察到不同的形式产生了捕捉直观动作特性的表示。我们还设计了各种基线模型用于动作选择和动作指定。尽管任务具有直观性，但我们观察到模型困难重重，人类表现明显优于它们。本研究旨在为未来的努力奠定基础，展示了连接视频理解中动作和效果的灵活性和多功能性，希望能激发出高级形式和模型的灵感。

    In this work, we introduce the novel concept of visually Connecting Actions and Their Effects (CATE) in video understanding. CATE can have applications in areas like task planning and learning from demonstration. We propose different CATE-based task formulations, such as action selection and action specification, where video understanding models connect actions and effects at semantic and fine-grained levels. We observe that different formulations produce representations capturing intuitive action properties. We also design various baseline models for action selection and action specification. Despite the intuitive nature of the task, we observe that models struggle, and humans outperform them by a large margin. The study aims to establish a foundation for future efforts, showcasing the flexibility and versatility of connecting actions and effects in video understanding, with the hope of inspiring advanced formulations and models.
    
[^8]: 自适应自训练框架用于细粒度场景图生成

    Adaptive Self-training Framework for Fine-grained Scene Graph Generation. (arXiv:2401.09786v1 [cs.CV])

    [http://arxiv.org/abs/2401.09786](http://arxiv.org/abs/2401.09786)

    本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。

    

    场景图生成（SGG）模型在基准数据集中存在长尾谓词分布和缺失注释问题。本研究旨在通过利用未标注的三元组缓解SGG的长尾问题。为此，我们引入了一种称为自训练SGG（ST-SGG）的框架，该框架基于未标注的三元组为其分配伪标签以训练SGG模型。虽然在图像识别方面的自训练取得了显著进展，但设计适用于SGG任务的自训练框架更具挑战，因为其固有特性，如语义歧义和长尾分布的谓词类别。因此，我们提出了一种新颖的SGG伪标签技术，称为具有动量的类别自适应阈值化（CATM），它是一种独立于模型的框架，可应用于任何已有的SGG模型。此外，我们设计了一个图结构学习器（GSL），从中获益。

    Scene graph generation (SGG) models have suffered from inherent problems regarding the benchmark datasets such as the long-tailed predicate distribution and missing annotation problems. In this work, we aim to alleviate the long-tailed problem of SGG by utilizing unannotated triplets. To this end, we introduce a Self-Training framework for SGG (ST-SGG) that assigns pseudo-labels for unannotated triplets based on which the SGG models are trained. While there has been significant progress in self-training for image recognition, designing a self-training framework for the SGG task is more challenging due to its inherent nature such as the semantic ambiguity and the long-tailed distribution of predicate classes. Hence, we propose a novel pseudo-labeling technique for SGG, called Class-specific Adaptive Thresholding with Momentum (CATM), which is a model-agnostic framework that can be applied to any existing SGG models. Furthermore, we devise a graph structure learner (GSL) that is benefici
    
[^9]: 观点文本倒置：通过预训练的2D扩散模型释放新颖的视图合成

    Viewpoint Textual Inversion: Unleashing Novel View Synthesis with Pretrained 2D Diffusion Models. (arXiv:2309.07986v1 [cs.CV])

    [http://arxiv.org/abs/2309.07986](http://arxiv.org/abs/2309.07986)

    本研究展示了通过预训练的2D图像扩散模型，可以从仅有2D监督的情况下提取出3D结构信息，并利用该信息进行3D视觉任务。通过观点神经文本倒置（ViewNeTI）方法，我们可以控制生成图像中对象的3D视点，有效解决新颖视图合成问题，并在单视图情况下具有良好的语义细节和逼真度。

    

    文本到图像扩散模型可以理解对象之间的空间关系，但它们是否能够仅通过2D监督来表示世界的真实3D结构？我们证明，是的，3D知识被编码在2D图像扩散模型（如稳定扩散模型）中，我们展示了这种结构可以用于3D视觉任务。我们的方法，观点神经文本倒置（ViewNeTI），可以控制生成图像中对象的3D视点。我们训练一个小型神经映射器，用于获取相机视点参数并预测文本编码器的潜在向量；然后利用这些潜在向量来调整扩散生成过程，生成具有所需相机视点的图像。ViewNeTI自然解决了新颖视图合成（NVS）问题。通过利用被冻结的扩散模型作为先验知识，我们可以用很少的输入视图来解决NVS问题；我们甚至可以进行单视图新颖视图合成。与之前的方法相比，我们的单视图NVS预测具有良好的语义细节和逼真度。

    Text-to-image diffusion models understand spatial relationship between objects, but do they represent the true 3D structure of the world from only 2D supervision? We demonstrate that yes, 3D knowledge is encoded in 2D image diffusion models like Stable Diffusion, and we show that this structure can be exploited for 3D vision tasks. Our method, Viewpoint Neural Textual Inversion (ViewNeTI), controls the 3D viewpoint of objects in generated images from frozen diffusion models. We train a small neural mapper to take camera viewpoint parameters and predict text encoder latents; the latents then condition the diffusion generation process to produce images with the desired camera viewpoint.  ViewNeTI naturally addresses Novel View Synthesis (NVS). By leveraging the frozen diffusion model as a prior, we can solve NVS with very few input views; we can even do single-view novel view synthesis. Our single-view NVS predictions have good semantic details and photorealism compared to prior methods.
    
[^10]: 建设多样化数字历史的高效OCR

    Efficient OCR for Building a Diverse Digital History. (arXiv:2304.02737v1 [cs.CV])

    [http://arxiv.org/abs/2304.02737](http://arxiv.org/abs/2304.02737)

    本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。

    

    每天有成千上万的用户查阅数字档案，但他们可以使用的信息并不能代表各种文献史料的多样性。典型用于光学字符识别（OCR）的序列到序列架构——联合学习视觉和语言模型——在低资源文献集合中很难扩展，因为学习语言-视觉模型需要大量标记的序列和计算。本研究将OCR建模为字符级图像检索问题，使用对比训练的视觉编码器。因为该模型只学习字符的视觉特征，它比现有架构更具样本效率和可扩展性，能够在现有解决方案失败的情况下实现准确的OCR。关键是，该模型为社区参与在使数字历史更具代表性的文献史料方面开辟了新的途径。

    Thousands of users consult digital archives daily, but the information they can access is unrepresentative of the diversity of documentary history. The sequence-to-sequence architecture typically used for optical character recognition (OCR) - which jointly learns a vision and language model - is poorly extensible to low-resource document collections, as learning a language-vision model requires extensive labeled sequences and compute. This study models OCR as a character level image retrieval problem, using a contrastively trained vision encoder. Because the model only learns characters' visual features, it is more sample efficient and extensible than existing architectures, enabling accurate OCR in settings where existing solutions fail. Crucially, the model opens new avenues for community engagement in making digital history more representative of documentary history.
    
[^11]: Point-MA2E:自监督点云学习的掩膜和仿射变换自编码器

    Point-MA2E: Masked and Affine Transformed AutoEncoder for Self-supervised Point Cloud Learning. (arXiv:2211.06841v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.06841](http://arxiv.org/abs/2211.06841)

    本文介绍了一种点云学习的自监督方法Point-MA2E，通过同时采用掩膜和仿射变换策略，实现了从损坏点云到还原点云的重建，扩展了目前掩膜方法的不足。

    

    在自监督点云学习中，掩膜建模已经证明了其有效性，通过从其掩膜对应部分重建完整点云。考虑到掩膜只会损坏输入的一部分点，本文推广仿射变换策略，通过特定规则破坏所有输入点，以补充流行的掩膜策略，从而实现点云学习的掩膜和仿射变换自编码器（Point-MA2E）。在此研究中，我们对点云进行仿射变换和掩膜，使用编码器-解码器模型从其损坏版本中重建原始点云。探索了各种点云编码器。对于非Transformer编码器，按照常见做法直接重建未损坏的点云。对于基于Transformer的编码器，我们将重建完整点云分解为详细的局部补丁和粗略的全局形状的重建。

    Masked modeling has demonstrated its effectiveness in self-supervised point cloud learning by reconstructing the complete point cloud from its masked counterpart. Considering that masking only corrupts partial points of the input, in this paper, we promote the affine transformation, which corrupts all input points with certain rules, to complement the popular masking strategy, leading to the Masked and Affine transformed AutoEncoder for point cloud learning (Point-MA2E). Generally, we corrupt the point cloud with affine transformation and masking as input and learn an encoder-decoder model to reconstruct the original point cloud from its corrupted version. Various point cloud encoders are explored in this study. For non-Transformer encoders, we follow the common practice to reconstruct the uncorrupted point cloud directly. For Transformer-based encoders, we decompose the reconstruction of the complete point cloud into the reconstructions of detailed local patches and rough global shape
    

