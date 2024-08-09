# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM](https://arxiv.org/abs/2403.20183) | HARMamba利用更轻量级的选择性SSM作为基础模型架构，以解决计算资源挑战 |
| [^2] | [FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression](https://arxiv.org/abs/2403.16677) | FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。 |
| [^3] | [P2LHAP:Wearable sensor-based human activity recognition, segmentation and forecast through Patch-to-Label Seq2Seq Transformer](https://arxiv.org/abs/2403.08214) | P2LHAP提出了一种新颖的Patch-to-Label Seq2Seq框架，可以在一个高效的单一任务模型中同时实现人类活动的分割、识别和预测 |
| [^4] | [CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering](https://arxiv.org/abs/2403.00816) | 该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。 |
| [^5] | [Towards Privacy-Aware Sign Language Translation at Scale](https://arxiv.org/abs/2402.09611) | 本研究提出了一种两阶段框架，用于实现规模化隐私感知手语翻译。我们利用自监督视频预训练和有监督微调的方法，在数据稀缺和隐私风险的情况下实现了最先进的手语翻译性能。 |
| [^6] | [Harmonized Spatial and Spectral Learning for Robust and Generalized Medical Image Segmentation.](http://arxiv.org/abs/2401.10373) | 本文提出了一种鲁棒且具普适性的医学图像分割方法，通过协调空间和光谱表示，引入光谱相关系数目标来提高对中阶特征和上下文长程依赖的捕捉能力，从而显著增强了泛化能力。 |
| [^7] | [Safe DreamerV3: Safe Reinforcement Learning with World Models.](http://arxiv.org/abs/2307.07176) | Safe DreamerV3是一种通过集成基于拉格朗日和计划的方法到世界模型中的新算法，实现了在低维度和仅采用视觉的任务中几乎零成本的安全强化学习。 |
| [^8] | [Loss Functions and Metrics in Deep Learning. A Review.](http://arxiv.org/abs/2307.02694) | 本文回顾了深度学习中最常见的损失函数和性能测量方法，旨在帮助从业者选择最适合其特定任务的方法。 |
| [^9] | [Improving Automated Hemorrhage Detection in Sparse-view Computed Tomography via Deep Convolutional Neural Network based Artifact Reduction.](http://arxiv.org/abs/2303.09340) | 本文提出了一种基于深度卷积神经网络的伪影降噪方法，用于改善稀疏视图下自动出血检测的图像质量，并证明其能够与完全采样的图像进行同等精确度的分类和检测。 |

# 详细

[^1]: HARMamba: 基于双向选择性SSM的高效可穿戴传感器人体活动识别

    HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM

    [https://arxiv.org/abs/2403.20183](https://arxiv.org/abs/2403.20183)

    HARMamba利用更轻量级的选择性SSM作为基础模型架构，以解决计算资源挑战

    

    可穿戴传感器的人体活动识别（HAR）是活动感知领域的重要研究领域。最近，一种高效的硬件感知状态空间模型（SSM）Mamba作为一种有前途的替代方案出现。HARMamba引入了更轻量级的选择性SSM作为活动识别的基本模型架构，以解决系统计算负载和内存使用的挑战。

    arXiv:2403.20183v1 Announce Type: cross  Abstract: Wearable sensor human activity recognition (HAR) is a crucial area of research in activity sensing. While transformer-based temporal deep learning models have been extensively studied and implemented, their large number of parameters present significant challenges in terms of system computing load and memory usage, rendering them unsuitable for real-time mobile activity recognition applications. Recently, an efficient hardware-aware state space model (SSM) called Mamba has emerged as a promising alternative. Mamba demonstrates strong potential in long sequence modeling, boasts a simpler network architecture, and offers an efficient hardware-aware design. Leveraging SSM for activity recognition represents an appealing avenue for exploration. In this study, we introduce HARMamba, which employs a more lightweight selective SSM as the foundational model architecture for activity recognition. The goal is to address the computational resourc
    
[^2]: FOOL: 用神经特征压缩解决卫星计算中的下行瓶颈问题

    FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression

    [https://arxiv.org/abs/2403.16677](https://arxiv.org/abs/2403.16677)

    FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。

    

    具有传感器的纳卫星星座捕获大范围地理区域，为地球观测提供了前所未有的机会。随着星座规模的增加，网络争用形成了下行瓶颈。轨道边缘计算（OEC）利用有限的机载计算资源通过在源头处理原始捕获来减少传输成本。然而，由于依赖粗糙的过滤方法或过分优先考虑特定下游任务，目前的解决方案具有有限的实用性。本文提出了FOOL，一种OEC本地和任务不可知的特征压缩方法，可保留预测性能。FOOL将高分辨率卫星图像进行分区，以最大化吞吐量。此外，它嵌入上下文并利用瓷砖间的依赖关系，以较低的开销降低传输成本。虽然FOOL是一种特征压缩器，但它可以在低

    arXiv:2403.16677v1 Announce Type: new  Abstract: Nanosatellite constellations equipped with sensors capturing large geographic regions provide unprecedented opportunities for Earth observation. As constellation sizes increase, network contention poses a downlink bottleneck. Orbital Edge Computing (OEC) leverages limited onboard compute resources to reduce transfer costs by processing the raw captures at the source. However, current solutions have limited practicability due to reliance on crude filtering methods or over-prioritizing particular downstream tasks.   This work presents FOOL, an OEC-native and task-agnostic feature compression method that preserves prediction performance. FOOL partitions high-resolution satellite imagery to maximize throughput. Further, it embeds context and leverages inter-tile dependencies to lower transfer costs with negligible overhead. While FOOL is a feature compressor, it can recover images with competitive scores on perceptual quality measures at low
    
[^3]: P2LHAP：基于可穿戴传感器的人类活动识别、分割和预测的Patch-to-Label Seq2Seq Transformer

    P2LHAP:Wearable sensor-based human activity recognition, segmentation and forecast through Patch-to-Label Seq2Seq Transformer

    [https://arxiv.org/abs/2403.08214](https://arxiv.org/abs/2403.08214)

    P2LHAP提出了一种新颖的Patch-to-Label Seq2Seq框架，可以在一个高效的单一任务模型中同时实现人类活动的分割、识别和预测

    

    传统深度学习方法很难同时从传感器数据中分割、识别和预测人类活动，限制了它们在医疗保健和辅助生活等领域的实用性，而这些领域对于实时理解正在进行和即将发生的活动至关重要。本文提出了P2LHAP，一种新颖的Patch-to-Label Seq2Seq框架，可以在一个高效的单一任务模型中解决这三个任务。P2LHAP将传感器数据流划分为一系列“补丁”，作为输入标记，并输出一系列包括预测的未来活动在内的补丁级活动标签。提出了一种基于周围补丁标签的独特平滑技术，可准确识别活动边界。此外，P2LHAP通过传感器信号通道独立的Transformer编码器和解码器学习补丁级表示。所有通道在所有序列上共享嵌入和Transformer权重。

    arXiv:2403.08214v1 Announce Type: cross  Abstract: Traditional deep learning methods struggle to simultaneously segment, recognize, and forecast human activities from sensor data. This limits their usefulness in many fields such as healthcare and assisted living, where real-time understanding of ongoing and upcoming activities is crucial. This paper introduces P2LHAP, a novel Patch-to-Label Seq2Seq framework that tackles all three tasks in a efficient single-task model. P2LHAP divides sensor data streams into a sequence of "patches", served as input tokens, and outputs a sequence of patch-level activity labels including the predicted future activities. A unique smoothing technique based on surrounding patch labels, is proposed to identify activity boundaries accurately. Additionally, P2LHAP learns patch-level representation by sensor signal channel-independent Transformer encoders and decoders. All channels share embedding and Transformer weights across all sequences. Evaluated on thre
    
[^4]: CFRet-DVQA：粗到精检索和高效调优用于文档视觉问答

    CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering

    [https://arxiv.org/abs/2403.00816](https://arxiv.org/abs/2403.00816)

    该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。

    

    文档视觉问答（DVQA）是一个涉及根据图像内容回答查询的任务。现有工作仅限于定位单页内的信息，不支持跨页面问答交互。此外，对模型输入的标记长度限制可能导致与答案相关的部分被截断。在本研究中，我们引入了一种简单但有效的方法学，称为CFRet-DVQA，重点放在检索和高效调优上，以有效解决这一关键问题。为此，我们首先从文档中检索与所提问题相关的多个片段。随后，我们利用大型语言模型（LLM）的先进推理能力，通过指导调优进一步增强其性能。该方法使得生成的答案与文档标签的风格相符。实验演示了...

    arXiv:2403.00816v1 Announce Type: cross  Abstract: Document Visual Question Answering (DVQA) is a task that involves responding to queries based on the content of images. Existing work is limited to locating information within a single page and does not facilitate cross-page question-and-answer interaction. Furthermore, the token length limitation imposed on inputs to the model may lead to truncation of segments pertinent to the answer. In this study, we introduce a simple but effective methodology called CFRet-DVQA, which focuses on retrieval and efficient tuning to address this critical issue effectively. For that, we initially retrieve multiple segments from the document that correlate with the question at hand. Subsequently, we leverage the advanced reasoning abilities of the large language model (LLM), further augmenting its performance through instruction tuning. This approach enables the generation of answers that align with the style of the document labels. The experiments demo
    
[^5]: 实现规模化隐私感知手语翻译

    Towards Privacy-Aware Sign Language Translation at Scale

    [https://arxiv.org/abs/2402.09611](https://arxiv.org/abs/2402.09611)

    本研究提出了一种两阶段框架，用于实现规模化隐私感知手语翻译。我们利用自监督视频预训练和有监督微调的方法，在数据稀缺和隐私风险的情况下实现了最先进的手语翻译性能。

    

    手语翻译的一个主要障碍是数据稀缺。目前在网络上可用的大部分手语数据由于缺乏对齐的字幕而无法用于训练监督模型。此外，使用大规模网络抓取的数据集来扩展手语翻译存在隐私风险，因为其中包含生物特征信息，负责任地开发手语翻译技术应该考虑这一点。在这项工作中，我们提出了一种针对规模化隐私感知手语翻译的两阶段框架，解决了这两个问题。我们引入了SSVP-SLT，它利用匿名和未注释的视频进行自监督视频预训练，然后利用经过筛选的平行数据集进行有监督的手语翻译微调。 SSVP-SLT在How2Sign数据集上实现了最新的微调和零次gloss-free手语翻译性能，比最强的基线模型提高了3个BLEU-4。通过受控实验，我们证明了我们的方法在多个语言和手语词汇上都具有较好的泛化能力。

    arXiv:2402.09611v1 Announce Type: new  Abstract: A major impediment to the advancement of sign language translation (SLT) is data scarcity. Much of the sign language data currently available on the web cannot be used for training supervised models due to the lack of aligned captions. Furthermore, scaling SLT using large-scale web-scraped datasets bears privacy risks due to the presence of biometric information, which the responsible development of SLT technologies should account for. In this work, we propose a two-stage framework for privacy-aware SLT at scale that addresses both of these issues. We introduce SSVP-SLT, which leverages self-supervised video pretraining on anonymized and unannotated videos, followed by supervised SLT finetuning on a curated parallel dataset. SSVP-SLT achieves state-of-the-art finetuned and zero-shot gloss-free SLT performance on the How2Sign dataset, outperforming the strongest respective baselines by over 3 BLEU-4. Based on controlled experiments, we fu
    
[^6]: 鲁棒且具普适性的医学图像分割的空间和光谱学习的协调

    Harmonized Spatial and Spectral Learning for Robust and Generalized Medical Image Segmentation. (arXiv:2401.10373v1 [eess.IV])

    [http://arxiv.org/abs/2401.10373](http://arxiv.org/abs/2401.10373)

    本文提出了一种鲁棒且具普适性的医学图像分割方法，通过协调空间和光谱表示，引入光谱相关系数目标来提高对中阶特征和上下文长程依赖的捕捉能力，从而显著增强了泛化能力。

    

    深度学习在医学图像分割方面取得了显著的成就。然而，由于类内变异性和类间独立性，现有的深度学习模型在泛化能力上存在困难，同一类在不同样本中表现不同，难以捕捉不同对象之间的复杂关系，从而导致更高的错误负例。本文提出了一种新的方法，通过协调空间和光谱表示来增强领域通用的医学图像分割。我们引入了创新的光谱相关系数目标，以提高模型捕捉中阶特征和上下文长程依赖的能力。这个目标通过融入有价值的光谱信息来补充传统的空间目标。大量实验证明，优化这个目标与现有的UNet和TransUNet架构显著提高了泛化能力。

    Deep learning has demonstrated remarkable achievements in medical image segmentation. However, prevailing deep learning models struggle with poor generalization due to (i) intra-class variations, where the same class appears differently in different samples, and (ii) inter-class independence, resulting in difficulties capturing intricate relationships between distinct objects, leading to higher false negative cases. This paper presents a novel approach that synergies spatial and spectral representations to enhance domain-generalized medical image segmentation. We introduce the innovative Spectral Correlation Coefficient objective to improve the model's capacity to capture middle-order features and contextual long-range dependencies. This objective complements traditional spatial objectives by incorporating valuable spectral information. Extensive experiments reveal that optimizing this objective with existing architectures like UNet and TransUNet significantly enhances generalization, 
    
[^7]: Safe DreamerV3：带有世界模型的安全强化学习

    Safe DreamerV3: Safe Reinforcement Learning with World Models. (arXiv:2307.07176v1 [cs.LG])

    [http://arxiv.org/abs/2307.07176](http://arxiv.org/abs/2307.07176)

    Safe DreamerV3是一种通过集成基于拉格朗日和计划的方法到世界模型中的新算法，实现了在低维度和仅采用视觉的任务中几乎零成本的安全强化学习。

    

    强化学习在真实世界场景中的广泛应用还没有实现, 这主要是因为其未能满足这些系统的基本安全需求。现有的安全强化学习方法使用成本函数来增强安全性，在复杂场景中，包括仅采用视觉的任务中，即使进行全面的数据采样和训练，也无法实现零成本。为了解决这个问题，我们引入了Safe DreamerV3，这是一种将基于拉格朗日和计划的方法集成到世界模型中的新算法。我们的方法论在SafeRL中代表了一个重要的进步，是第一个在Safety-Gymnasium基准中实现近乎零成本的算法。我们的项目网站可以在以下链接找到：https://sites.google.com/view/safedreamerv3。

    The widespread application of Reinforcement Learning (RL) in real-world situations is yet to come to fruition, largely as a result of its failure to satisfy the essential safety demands of such systems. Existing safe reinforcement learning (SafeRL) methods, employing cost functions to enhance safety, fail to achieve zero-cost in complex scenarios, including vision-only tasks, even with comprehensive data sampling and training. To address this, we introduce Safe DreamerV3, a novel algorithm that integrates both Lagrangian-based and planning-based methods within a world model. Our methodology represents a significant advancement in SafeRL as the first algorithm to achieve nearly zero-cost in both low-dimensional and vision-only tasks within the Safety-Gymnasium benchmark. Our project website can be found in: https://sites.google.com/view/safedreamerv3.
    
[^8]: 深度学习中的损失函数和度量方法：一项评论

    Loss Functions and Metrics in Deep Learning. A Review. (arXiv:2307.02694v1 [cs.LG])

    [http://arxiv.org/abs/2307.02694](http://arxiv.org/abs/2307.02694)

    本文回顾了深度学习中最常见的损失函数和性能测量方法，旨在帮助从业者选择最适合其特定任务的方法。

    

    深度学习的一个重要组成部分是选择用于训练和评估模型的损失函数和性能度量。本文回顾了深度学习中最常见的损失函数和性能测量方法。我们探讨了每种技术的优势和局限性，并举例说明它们在各种深度学习问题上的应用。我们的评论旨在全面了解最常见的深度学习任务中使用的不同损失函数和性能指标，并帮助从业者选择最适合其特定任务的方法。

    One of the essential components of deep learning is the choice of the loss function and performance metrics used to train and evaluate models. This paper reviews the most prevalent loss functions and performance measurements in deep learning. We examine the benefits and limits of each technique and illustrate their application to various deep-learning problems. Our review aims to give a comprehensive picture of the different loss functions and performance indicators used in the most common deep learning tasks and help practitioners choose the best method for their specific task.
    
[^9]: 基于深度卷积神经网络伪影降噪的稀疏视图CT图像自动出血检测的改进

    Improving Automated Hemorrhage Detection in Sparse-view Computed Tomography via Deep Convolutional Neural Network based Artifact Reduction. (arXiv:2303.09340v1 [eess.IV])

    [http://arxiv.org/abs/2303.09340](http://arxiv.org/abs/2303.09340)

    本文提出了一种基于深度卷积神经网络的伪影降噪方法，用于改善稀疏视图下自动出血检测的图像质量，并证明其能够与完全采样的图像进行同等精确度的分类和检测。

    

    颅内出血是一种严重的健康问题，需要快速且常常非常密集的医疗治疗。为了诊断，通常要进行颅部计算机断层扫描（CCT）扫描。然而，由于辐射引起的增加的健康风险是一个问题。降低这种潜在风险的最重要策略是尽可能保持辐射剂量低，并与诊断任务一致。 稀疏视图CT可以通过减少所采集的视图总数，从而降低剂量，是一种有效的策略，但代价是降低图像质量。在这项工作中，我们使用U-Net架构来减少稀疏视图CCT的伪影，从稀疏视图中预测完全采样的重建图像。我们使用一个卷积神经网络对出血的检测和分类进行评估，并在完全采样的CCT上进行训练。我们的结果表明，伪影降噪后的CCT图像进行自动分类和检测的准确性与完全采样的CCT图像没有明显差异。

    Intracranial hemorrhage poses a serious health problem requiring rapid and often intensive medical treatment. For diagnosis, a Cranial Computed Tomography (CCT) scan is usually performed. However, the increased health risk caused by radiation is a concern. The most important strategy to reduce this potential risk is to keep the radiation dose as low as possible and consistent with the diagnostic task. Sparse-view CT can be an effective strategy to reduce dose by reducing the total number of views acquired, albeit at the expense of image quality. In this work, we use a U-Net architecture to reduce artifacts from sparse-view CCTs, predicting fully sampled reconstructions from sparse-view ones. We evaluate the hemorrhage detectability in the predicted CCTs with a hemorrhage classification convolutional neural network, trained on fully sampled CCTs to detect and classify different sub-types of hemorrhages. Our results suggest that the automated classification and detection accuracy of hemo
    

