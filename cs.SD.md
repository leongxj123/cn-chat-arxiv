# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Listenable Maps for Audio Classifiers](https://arxiv.org/abs/2403.13086) | 引入了一种名为Listenable Maps for Audio Classifiers (L-MAC)的可听图方法，用于生成忠实且可听的音频分类器解释。 |
| [^2] | [Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach.](http://arxiv.org/abs/2401.10747) | 本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。 |
| [^3] | [CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models.](http://arxiv.org/abs/2310.08753) | CompA提出了由两个专家注释的音频-语言模型组合推理基准数据集，用于评估ALMs在理解音频中声音事件的顺序和属性绑定方面的表现。 |
| [^4] | [COVID-19 Detection System: A Comparative Analysis of System Performance Based on Acoustic Features of Cough Audio Signals.](http://arxiv.org/abs/2309.04505) | 本研究通过比较分析声学特征对咳嗽音频信号的机器学习模型性能的影响，提出了一种高效的COVID-19检测系统。 |

# 详细

[^1]: 可听图用于音频分类器

    Listenable Maps for Audio Classifiers

    [https://arxiv.org/abs/2403.13086](https://arxiv.org/abs/2403.13086)

    引入了一种名为Listenable Maps for Audio Classifiers (L-MAC)的可听图方法，用于生成忠实且可听的音频分类器解释。

    

    尽管深度学习模型在各种任务上表现出色，其复杂性给解释提出了挑战。这一挑战在音频信号中尤为明显，传达解释变得困难。为解决这一问题，我们引入了用于音频分类器的可听图（Listenable Maps for Audio Classifiers，L-MAC），这是一种生成忠实且可听解释的后处理解释方法。L-MAC利用预训练分类器之上的解码器生成二值掩码，突出显示输入音频的相关部分。我们用一种特殊损失来训练解码器，该损失最大化分类器对输入音频的掩码部分的置信度，同时最小化模型对掩码部分输出的概率。对领域内和领域外数据的定量评估表明，L-MAC始终产生比几种梯度和掩码方法更忠实的解释。

    arXiv:2403.13086v1 Announce Type: cross  Abstract: Despite the impressive performance of deep learning models across diverse tasks, their complexity poses challenges for interpretation. This challenge is particularly evident for audio signals, where conveying interpretations becomes inherently difficult. To address this issue, we introduce Listenable Maps for Audio Classifiers (L-MAC), a posthoc interpretation method that generates faithful and listenable interpretations. L-MAC utilizes a decoder on top of a pretrained classifier to generate binary masks that highlight relevant portions of the input audio. We train the decoder with a special loss that maximizes the confidence of the classifier decision on the masked-in portion of the audio while minimizing the probability of model output for the masked-out portion. Quantitative evaluations on both in-domain and out-of-domain data demonstrate that L-MAC consistently produces more faithful interpretations than several gradient and maskin
    
[^2]: 缺失模态下的多模态情感分析:一种知识迁移方法

    Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach. (arXiv:2401.10747v1 [cs.SD])

    [http://arxiv.org/abs/2401.10747](http://arxiv.org/abs/2401.10747)

    本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。

    

    多模态情感分析旨在通过视觉、语言和声音线索来识别个体表达的情绪。然而，现有研究大多假设在训练和测试过程中所有模态都是可用的，这使得它们的算法容易受到缺失模态的影响。在本文中，我们提出了一种新颖的知识迁移网络，用于在不同模态之间进行翻译，以重构缺失的音频模态。此外，我们还开发了一种跨模态注意机制，以保留重构和观察到的模态的最大信息，用于情感预测。在三个公开数据集上进行的大量实验证明了相对于基线算法的显著改进，并实现了与具有完整多模态监督的先前方法相媲美的结果。

    Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most of the existing research efforts assume that all modalities are available during both training and testing, making their algorithms susceptible to the missing modality scenario. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio modalities. Moreover, we develop a cross-modality attention mechanism to retain the maximal information of the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baselines and achieve comparable results to the previous methods with complete multi-modality supervision.
    
[^3]: CompA: 解决音频-语言模型中的组合推理差距

    CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models. (arXiv:2310.08753v1 [cs.SD])

    [http://arxiv.org/abs/2310.08753](http://arxiv.org/abs/2310.08753)

    CompA提出了由两个专家注释的音频-语言模型组合推理基准数据集，用于评估ALMs在理解音频中声音事件的顺序和属性绑定方面的表现。

    

    音频的基本特性是其组合性。使用对比方法（例如CLAP）训练的音频-语言模型（ALMs）能够学习音频和语言模态之间的共享表示，从而在许多下游应用中提高性能，包括零样本音频分类、音频检索等。然而，这些模型在有效执行组合推理方面的能力还很少被探索，需要进一步的研究。本文提出了CompA，这是一个由两个专家注释的基准数据集，其中大多数是真实世界的音频样本，用于评估ALMs的组合推理能力。我们的CompA-order评估ALMs在理解音频中声音事件的顺序或发生时的表现如何，而CompA-attribute评估声音事件的属性绑定。每个基准数据集中的实例包含两个音频-标题对，其中两个音频具有相同的声音事件，但组合方式不同。

    A fundamental characteristic of audio is its compositional nature. Audio-language models (ALMs) trained using a contrastive approach (e.g., CLAP) that learns a shared representation between audio and language modalities have improved performance in many downstream applications, including zero-shot audio classification, audio retrieval, etc. However, the ability of these models to effectively perform compositional reasoning remains largely unexplored and necessitates additional research. In this paper, we propose CompA, a collection of two expert-annotated benchmarks with a majority of real-world audio samples, to evaluate compositional reasoning in ALMs. Our proposed CompA-order evaluates how well an ALM understands the order or occurrence of acoustic events in audio, and CompA-attribute evaluates attribute binding of acoustic events. An instance from either benchmark consists of two audio-caption pairs, where both audios have the same acoustic events but with different compositions. A
    
[^4]: COVID-19检测系统：基于咳嗽音频信号的声学特征系统性能比较分析

    COVID-19 Detection System: A Comparative Analysis of System Performance Based on Acoustic Features of Cough Audio Signals. (arXiv:2309.04505v1 [cs.SD])

    [http://arxiv.org/abs/2309.04505](http://arxiv.org/abs/2309.04505)

    本研究通过比较分析声学特征对咳嗽音频信号的机器学习模型性能的影响，提出了一种高效的COVID-19检测系统。

    

    各种呼吸道疾病如感冒和流感、哮喘以及COVID-19等在全球范围内影响着人们的日常生活。在医学实践中，呼吸声音被广泛用于医疗服务中，用于诊断各种呼吸系统疾病和肺部疾病。传统的诊断方法需要专业知识，成本高且依赖于人类专业知识。最近，咳嗽音频记录被用来自动化检测呼吸系统疾病的过程。本研究旨在检查各种声学特征，以提高机器学习模型在咳嗽信号中检测COVID-19的性能。本研究调查了三种特征提取技术（MFCC，Chroma和Spectral Contrast特征）在两种机器学习算法（SVM和MLP）上的功效，并提出了一种高效的COVID-19检测系统。

    A wide range of respiratory diseases, such as cold and flu, asthma, and COVID-19, affect people's daily lives worldwide. In medical practice, respiratory sounds are widely used in medical services to diagnose various respiratory illnesses and lung disorders. The traditional diagnosis of such sounds requires specialized knowledge, which can be costly and reliant on human expertise. Recently, cough audio recordings have been used to automate the process of detecting respiratory conditions. This research aims to examine various acoustic features that enhance the performance of machine learning (ML) models in detecting COVID-19 from cough signals. This study investigates the efficacy of three feature extraction techniques, including Mel Frequency Cepstral Coefficients (MFCC), Chroma, and Spectral Contrast features, on two ML algorithms, Support Vector Machine (SVM) and Multilayer Perceptron (MLP), and thus proposes an efficient COVID-19 detection system. The proposed system produces a prac
    

