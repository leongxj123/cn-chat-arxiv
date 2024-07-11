# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Objective and subjective evaluation of speech enhancement methods in the UDASE task of the 7th CHiME challenge](https://rss.arxiv.org/abs/2402.01413) | 本文介绍了第七届CHiME挑战赛的UDASE任务中系统的客观和主观评估，并分析了结果 |
| [^2] | [Transforming LLMs into Cross-modal and Cross-lingual RetrievalSystems](https://arxiv.org/abs/2404.01616) | 提出使用LLMs初始化多模态DE检索系统，实现在102种语言中匹配语音和文本的能力，无需在LLM预训练期间使用语音数据，且相比先前系统取得10%的Recall@1绝对改进 |
| [^3] | [DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution](https://arxiv.org/abs/2305.17000) | DistriBlock提出了一种能够识别对抗性音频样本的有效检测策略，通过利用输出分布的特征，包括中位数、最大值和最小值、熵以及与后续时间步骤的分布之间的散度，应用二元分类器进行预测。这项研究证明了DistriBlock在识别对抗性音频样本方面的有效性。 |
| [^4] | [A noise-robust acoustic method for recognition of foraging activities of grazing cattle.](http://arxiv.org/abs/2304.14824) | 本研究提出了一种抗噪声的声学方法，能够分析与吃草和反刍相关的鉴定下颚运动事件的固定长度段，用于识别牛的觅食活动，并在环境和自然噪声方面具有鲁棒性。 |

# 详细

[^1]: 第七届CHiME挑战赛中UDASE任务中语音增强方法的客观和主观评估

    Objective and subjective evaluation of speech enhancement methods in the UDASE task of the 7th CHiME challenge

    [https://rss.arxiv.org/abs/2402.01413](https://rss.arxiv.org/abs/2402.01413)

    本文介绍了第七届CHiME挑战赛的UDASE任务中系统的客观和主观评估，并分析了结果

    

    基于监督模型的语音增强方法是通过人工合成的干净语音和噪声信号混合来训练的。然而，合成训练条件可能无法准确反映测试过程中遇到的真实世界条件。这种差异可能导致在测试域与合成训练域显著不同时性能不佳。为了解决这个问题，第七届CHiME挑战赛的UDASE任务旨在利用测试域的真实世界噪声语音录音来对语音增强模型进行无监督域适应。具体来说，这个测试域对应于CHiME-5数据集，该数据集由在嘈杂和混响的家庭环境中进行的真实多说话人对话录音组成，无法获得地面实况干净语音信号。在本文中，我们介绍了提交到CHiME-7 UDASE任务的系统的客观和主观评估，并对结果进行了分析

    Supervised models for speech enhancement are trained using artificially generated mixtures of clean speech and noise signals. However, the synthetic training conditions may not accurately reflect real-world conditions encountered during testing. This discrepancy can result in poor performance when the test domain significantly differs from the synthetic training domain. To tackle this issue, the UDASE task of the 7th CHiME challenge aimed to leverage real-world noisy speech recordings from the test domain for unsupervised domain adaptation of speech enhancement models. Specifically, this test domain corresponds to the CHiME-5 dataset, characterized by real multi-speaker and conversational speech recordings made in noisy and reverberant domestic environments, for which ground-truth clean speech signals are not available. In this paper, we present the objective and subjective evaluations of the systems that were submitted to the CHiME-7 UDASE task, and we provide an analysis of the resul
    
[^2]: 将LLMs转化为跨模态和跨语言检索系统

    Transforming LLMs into Cross-modal and Cross-lingual RetrievalSystems

    [https://arxiv.org/abs/2404.01616](https://arxiv.org/abs/2404.01616)

    提出使用LLMs初始化多模态DE检索系统，实现在102种语言中匹配语音和文本的能力，无需在LLM预训练期间使用语音数据，且相比先前系统取得10%的Recall@1绝对改进

    

    大型语言模型（LLMs）是在仅基于文本数据进行训练的，这超出了具有配对语音和文本数据的语言范围。同时，基于双编码器（DE）的检索系统将查询和文档投影到相同的嵌入空间中，并在检索和双语文本挖掘中展示了成功。为了在许多语言中匹配语音和文本，我们建议使用LLMs初始化多模态DE检索系统。与传统方法不同，我们的系统在LLM预训练期间不需要语音数据，并且可以利用LLM的多语言文本理解能力来匹配检索训练期间看不见的语言中的语音和文本。我们的多模态LLM-based检索系统能够在102种语言中匹配语音和文本，尽管只在21种语言上进行了训练。我们的系统优于先前专门在所有102种语言上训练的系统。在这些语言中，我们在Recall@1上实现了10％的绝对改进。

    arXiv:2404.01616v1 Announce Type: new  Abstract: Large language models (LLMs) are trained on text-only data that go far beyond the languages with paired speech and text data. At the same time, Dual Encoder (DE) based retrieval systems project queries and documents into the same embedding space and have demonstrated their success in retrieval and bi-text mining. To match speech and text in many languages, we propose using LLMs to initialize multi-modal DE retrieval systems. Unlike traditional methods, our system doesn't require speech data during LLM pre-training and can exploit LLM's multilingual text understanding capabilities to match speech and text in languages unseen during retrieval training. Our multi-modal LLM-based retrieval system is capable of matching speech and text in 102 languages despite only training on 21 languages. Our system outperforms previous systems trained explicitly on all 102 languages. We achieve a 10% absolute improvement in Recall@1 averaged across these l
    
[^3]: DistriBlock: 通过利用输出分布的特征识别对抗性音频样本

    DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution

    [https://arxiv.org/abs/2305.17000](https://arxiv.org/abs/2305.17000)

    DistriBlock提出了一种能够识别对抗性音频样本的有效检测策略，通过利用输出分布的特征，包括中位数、最大值和最小值、熵以及与后续时间步骤的分布之间的散度，应用二元分类器进行预测。这项研究证明了DistriBlock在识别对抗性音频样本方面的有效性。

    

    对抗性攻击可能误导自动语音识别（ASR）系统，使其预测任意目标文本，从而构成明显的安全威胁。为了防止这种攻击，我们提出了DistriBlock，一种适用于任何ASR系统的高效检测策略，该系统在每个时间步骤上预测输出标记的概率分布。我们对该分布的一组特征进行测量：输出概率的中位数、最大值和最小值，分布的熵，以及与后续时间步骤的分布之间的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性和对抗性数据观察到的特征，我们应用二元分类器，包括简单的基于阈值的分类、这种分类器的集合以及神经网络。通过对不同最先进的ASR系统和语言数据集进行广泛分析，我们证明了DistriBlock在识别对抗性音频样本方面的有效性。

    arXiv:2305.17000v2 Announce Type: replace-cross  Abstract: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, 
    
[^4]: 一种抗噪声的声学方法用于识别牛的觅食活动

    A noise-robust acoustic method for recognition of foraging activities of grazing cattle. (arXiv:2304.14824v1 [cs.LG])

    [http://arxiv.org/abs/2304.14824](http://arxiv.org/abs/2304.14824)

    本研究提出了一种抗噪声的声学方法，能够分析与吃草和反刍相关的鉴定下颚运动事件的固定长度段，用于识别牛的觅食活动，并在环境和自然噪声方面具有鲁棒性。

    

    为了在不断增长的乳制品市场中保持竞争力，农民必须不断改进他们的畜牧生产系统。精确畜牧业技术提供了商业农场动物个体化监测，优化畜牧生产。连续的声学监测是一种广泛接受的感应技术，用于估计自由放牧牛的日反刍和吃草时间预算。然而，牧场上的典型环境和自然噪声明显影响当前声学方法的性能和泛化。在本研究中，我们提出了一种声学方法，称为抗噪声觅食活动识别器 (NRFAR)。该方法通过分析与吃草和反刍相关的鉴定下颚运动事件的固定长度段，确定觅食活动的突发。NRFAR 的加性噪声鲁棒性使用静态高斯白噪声和四种不同的非静态自然噪声进行评估。

    To stay competitive in the growing dairy market, farmers must continuously improve their livestock production systems. Precision livestock farming technologies provide individualised monitoring of animals on commercial farms, optimising livestock production. Continuous acoustic monitoring is a widely accepted sensing technique used to estimate the daily rumination and grazing time budget of free-ranging cattle. However, typical environmental and natural noises on pasture noticeably affect the performance and generalisation of current acoustic methods. In this study, we present an acoustic method called Noise-Robust Foraging Activity Recognizer (NRFAR). The proposed method determines foraging activity bouts by analysing fixed-length segments of identified jaw movement events associated with grazing and rumination. The additive noise robustness of NRFAR was evaluated for several signal-to-noise ratios, using stationary Gaussian white noise and four different non-stationary natural noise 
    

