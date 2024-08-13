# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a Transformer](https://arxiv.org/abs/2403.20324) | 本研究通过引入Transformer模型结合跨通道注意力，推动了使用深度学习进行单脉冲电刺激响应的SOZ本地化，在评估中展示了模型对未见患者和电极放置的泛化能力 |
| [^2] | [Brain-grounding of semantic vectors improves neural decoding of visual stimuli](https://arxiv.org/abs/2403.15176) | 提出了一种表示学习框架，称为语义向量的脑接地，通过微调预训练的特征向量，使其更好地与人类大脑中视觉刺激的神经表示对齐。 |

# 详细

[^1]: 使用变压器从单脉冲电刺激响应中定位癫痫发作起始区

    Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a Transformer

    [https://arxiv.org/abs/2403.20324](https://arxiv.org/abs/2403.20324)

    本研究通过引入Transformer模型结合跨通道注意力，推动了使用深度学习进行单脉冲电刺激响应的SOZ本地化，在评估中展示了模型对未见患者和电极放置的泛化能力

    

    癫痫是最常见的神经疾病之一，许多患者在药物无法控制癫痫发作时需要手术干预。为了取得有效的手术结果，准确定位癫痫发作起始区 - 通常近似为癫痫发作起始区 (SOZ) - 至关重要但仍然具有挑战性。通过电刺激进行主动探测已经成为识别癫痫发作区域的标准临床实践。本文推动了深度学习在使用单脉冲电刺激 (SPES) 响应进行 SOZ 定位的应用。我们通过引入包含跨通道注意力的Transformer模型来实现这一点。我们在保留的患者测试集上评估这些模型，以评估它们对未见患者和电极放置的泛化能力。

    arXiv:2403.20324v1 Announce Type: new  Abstract: Epilepsy is one of the most common neurological disorders, and many patients require surgical intervention when medication fails to control seizures. For effective surgical outcomes, precise localisation of the epileptogenic focus - often approximated through the Seizure Onset Zone (SOZ) - is critical yet remains a challenge. Active probing through electrical stimulation is already standard clinical practice for identifying epileptogenic areas. This paper advances the application of deep learning for SOZ localisation using Single Pulse Electrical Stimulation (SPES) responses. We achieve this by introducing Transformer models that incorporate cross-channel attention. We evaluate these models on held-out patient test sets to assess their generalisability to unseen patients and electrode placements.   Our study makes three key contributions: Firstly, we implement an existing deep learning model to compare two SPES analysis paradigms - namel
    
[^2]: 语义向量的脑接地改善了神经解码视觉刺激

    Brain-grounding of semantic vectors improves neural decoding of visual stimuli

    [https://arxiv.org/abs/2403.15176](https://arxiv.org/abs/2403.15176)

    提出了一种表示学习框架，称为语义向量的脑接地，通过微调预训练的特征向量，使其更好地与人类大脑中视觉刺激的神经表示对齐。

    

    发展准确全面的算法来解码大脑内容是神经科学和脑机接口领域的一个长期目标。之前的研究已经证明了通过训练机器学习模型将大脑活动模式映射到一个语义向量表示的神经解码的可行性。为了解决这个问题，我们提出了一个表示学习框架，称为语义向量的脑接地，它对预训练的特征向量进行微调，以更好地与人类大脑中视觉刺激的神经表示对齐。

    arXiv:2403.15176v1 Announce Type: cross  Abstract: Developing algorithms for accurate and comprehensive neural decoding of mental contents is one of the long-cherished goals in the field of neuroscience and brain-machine interfaces. Previous studies have demonstrated the feasibility of neural decoding by training machine learning models to map brain activity patterns into a semantic vector representation of stimuli. These vectors, hereafter referred as pretrained feature vectors, are usually derived from semantic spaces based solely on image and/or text features and therefore they might have a totally different characteristics than how visual stimuli is represented in the human brain, resulting in limiting the capability of brain decoders to learn this mapping. To address this issue, we propose a representation learning framework, termed brain-grounding of semantic vectors, which fine-tunes pretrained feature vectors to better align with the neural representation of visual stimuli in t
    

