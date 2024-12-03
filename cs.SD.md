# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Proactive and Dual Prevention Mechanism against Illegal Song Covers empowered by Singing Voice Conversion.](http://arxiv.org/abs/2401.17133) | 这项工作提出了一种主动性的双重防护机制，通过引入人类无法察觉的扰动，干扰歌唱声音转换的生成过程，防止未经授权的基于歌唱声音转换的非法歌曲翻唱。该机制既扰乱了歌手身份，又扰乱了歌词，使得歌唱声音既不模仿目标歌手，也不保留原始歌词。 |
| [^2] | [Overlap-aware End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization.](http://arxiv.org/abs/2401.12850) | 本文提出了一种针对演讲者分割的端到端监督分层图聚类算法，使用图神经网络进行表示学习、度量学习和聚类，并通过外部重叠检测器提供额外的输入。 |
| [^3] | [Separate Anything You Describe.](http://arxiv.org/abs/2308.05037) | 这项工作介绍了一种用于开放领域音频源分离的基础模型AudioSep，该模型使用自然语言查询，具有强大的分离性能和优秀的泛化能力。 |

# 详细

[^1]: 一种针对非法歌曲翻唱的主动性双重防护机制：基于歌唱声音转换的能力

    A Proactive and Dual Prevention Mechanism against Illegal Song Covers empowered by Singing Voice Conversion. (arXiv:2401.17133v1 [cs.SD])

    [http://arxiv.org/abs/2401.17133](http://arxiv.org/abs/2401.17133)

    这项工作提出了一种主动性的双重防护机制，通过引入人类无法察觉的扰动，干扰歌唱声音转换的生成过程，防止未经授权的基于歌唱声音转换的非法歌曲翻唱。该机制既扰乱了歌手身份，又扰乱了歌词，使得歌唱声音既不模仿目标歌手，也不保留原始歌词。

    

    歌唱声音转换(SVC)通过将一个歌手的歌唱声音转换成另一个目标歌手的歌唱声音，并使用原始歌词和旋律，自动化了歌曲翻唱。然而，这引发了对版权和公民权利的严重担忧。本研究提出了 SongBsAb，这是第一个主动性方法，用于减轻未经授权的基于 SVC 的非法歌曲翻唱。SongBsAb 在发布歌唱声音之前引入了人类无法察觉的扰动，这样当它们被使用时，SVC 的生成过程将被干扰，导致意外的歌唱声音。 SongBsAb 具有双重预防效果，引起歌手身份和歌词的混乱，即 SVC 覆盖的歌唱声音既不模仿目标歌手，也不保留原始歌词。为了提高扰动的不可察觉性，我们使用了一个以伴奏曲作为额外掩蔽者的基于心理声学模型的损失模型。

    Singing voice conversion (SVC) automates song covers by converting one singer's singing voice into another target singer's singing voice with the original lyrics and melody. However, it raises serious concerns about copyright and civil right infringements to multiple entities. This work proposes SongBsAb, the first proactive approach to mitigate unauthorized SVC-based illegal song covers. SongBsAb introduces human-imperceptible perturbations to singing voices before releasing them, so that when they are used, the generation process of SVC will be interfered, resulting in unexpected singing voices. SongBsAb features a dual prevention effect by causing both (singer) identity disruption and lyric disruption, namely, the SVC-covered singing voice neither imitates the target singer nor preserves the original lyrics. To improve the imperceptibility of perturbations, we refine a psychoacoustic model-based loss with the backing track as an additional masker, a unique accompanying element for s
    
[^2]: 针对演讲者分割的端到端监督分层图聚类算法

    Overlap-aware End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization. (arXiv:2401.12850v1 [eess.AS])

    [http://arxiv.org/abs/2401.12850](http://arxiv.org/abs/2401.12850)

    本文提出了一种针对演讲者分割的端到端监督分层图聚类算法，使用图神经网络进行表示学习、度量学习和聚类，并通过外部重叠检测器提供额外的输入。

    

    演讲者分割是基于说话者身份对音频录音进行分割的重要语音预处理步骤，适用于多个下游应用。传统的分割方法涉及多次嵌入提取和聚类步骤，通常以孤立的方式进行优化。虽然端到端的分割系统试图学习一个单一模型来完成任务，但通常训练复杂且需要大量的监督数据集。在本文中，我们提出了一种基于图神经网络(GNN)的端到端监督分层聚类算法，称为E-SHARC。E-SHARC方法使用前端mel-filterbank特征作为输入，并联合学习嵌入提取器和GNN聚类模块，进行表示学习、度量学习和端到端优化的聚类。此外，E-SHARC还通过外部重叠检测器提供额外的输入。

    Speaker diarization, the task of segmenting an audio recording based on speaker identity, constitutes an important speech pre-processing step for several downstream applications. The conventional approach to diarization involves multiple steps of embedding extraction and clustering, which are often optimized in an isolated fashion. While end-to-end diarization systems attempt to learn a single model for the task, they are often cumbersome to train and require large supervised datasets. In this paper, we propose an end-to-end supervised hierarchical clustering algorithm based on graph neural networks (GNN), called End-to-end Supervised HierARchical Clustering (E-SHARC). The E-SHARC approach uses front-end mel-filterbank features as input and jointly learns an embedding extractor and the GNN clustering module, performing representation learning, metric learning, and clustering with end-to-end optimization. Further, with additional inputs from an external overlap detector, the E-SHARC app
    
[^3]: 将任何你描述的事物分离

    Separate Anything You Describe. (arXiv:2308.05037v1 [eess.AS])

    [http://arxiv.org/abs/2308.05037](http://arxiv.org/abs/2308.05037)

    这项工作介绍了一种用于开放领域音频源分离的基础模型AudioSep，该模型使用自然语言查询，具有强大的分离性能和优秀的泛化能力。

    

    语言查询音频源分离（LASS）是计算听觉场景分析（CASA）中的一种新范 Paradigm。LASS旨在根据自然语言查询从音频混合物中分离目标声音，为数字音频应用提供了一种自然且可扩展的界面。尽管最近在LASS上取得了有希望的分离性能（例如，乐器，有限类别的音频事件），但仍然无法在开放域中分离音频概念。在这项工作中，我们引入了AudioSep，这是一种针对自然语言查询的开放领域音频源分离的基础模型。我们使用大规模多模态数据集训练AudioSep，并对其在许多任务上进行了广泛评估，包括音频事件分离，乐器分离和语音增强。AudioSep表现出强大的分离性能和令人印象深刻的零-shot泛化能力，使用音频标题或文字标签作为查询，明显优于其他方法。

    Language-queried audio source separation (LASS) is a new paradigm for computational auditory scene analysis (CASA). LASS aims to separate a target sound from an audio mixture given a natural language query, which provides a natural and scalable interface for digital audio applications. Recent works on LASS, despite attaining promising separation performance on specific sources (e.g., musical instruments, limited classes of audio events), are unable to separate audio concepts in the open domain. In this work, we introduce AudioSep, a foundation model for open-domain audio source separation with natural language queries. We train AudioSep on large-scale multimodal datasets and extensively evaluate its capabilities on numerous tasks including audio event separation, musical instrument separation, and speech enhancement. AudioSep demonstrates strong separation performance and impressive zero-shot generalization ability using audio captions or text labels as queries, substantially outperfor
    

