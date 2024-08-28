# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a Transformer](https://arxiv.org/abs/2403.20324) | 本研究通过引入Transformer模型结合跨通道注意力，推动了使用深度学习进行单脉冲电刺激响应的SOZ本地化，在评估中展示了模型对未见患者和电极放置的泛化能力 |
| [^2] | [Predicting O-GlcNAcylation Sites in Mammalian Proteins with Transformers and RNNs Trained with a New Loss Function](https://arxiv.org/abs/2402.17131) | 本研究提出了一种新的损失函数，称为加权焦点可微MCC，用于改善分类模型的性能，并在预测哺乳动物蛋白质中的O-GlcNAcylation位点方面取得了进展 |

# 详细

[^1]: 使用变压器从单脉冲电刺激响应中定位癫痫发作起始区

    Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a Transformer

    [https://arxiv.org/abs/2403.20324](https://arxiv.org/abs/2403.20324)

    本研究通过引入Transformer模型结合跨通道注意力，推动了使用深度学习进行单脉冲电刺激响应的SOZ本地化，在评估中展示了模型对未见患者和电极放置的泛化能力

    

    癫痫是最常见的神经疾病之一，许多患者在药物无法控制癫痫发作时需要手术干预。为了取得有效的手术结果，准确定位癫痫发作起始区 - 通常近似为癫痫发作起始区 (SOZ) - 至关重要但仍然具有挑战性。通过电刺激进行主动探测已经成为识别癫痫发作区域的标准临床实践。本文推动了深度学习在使用单脉冲电刺激 (SPES) 响应进行 SOZ 定位的应用。我们通过引入包含跨通道注意力的Transformer模型来实现这一点。我们在保留的患者测试集上评估这些模型，以评估它们对未见患者和电极放置的泛化能力。

    arXiv:2403.20324v1 Announce Type: new  Abstract: Epilepsy is one of the most common neurological disorders, and many patients require surgical intervention when medication fails to control seizures. For effective surgical outcomes, precise localisation of the epileptogenic focus - often approximated through the Seizure Onset Zone (SOZ) - is critical yet remains a challenge. Active probing through electrical stimulation is already standard clinical practice for identifying epileptogenic areas. This paper advances the application of deep learning for SOZ localisation using Single Pulse Electrical Stimulation (SPES) responses. We achieve this by introducing Transformer models that incorporate cross-channel attention. We evaluate these models on held-out patient test sets to assess their generalisability to unseen patients and electrode placements.   Our study makes three key contributions: Firstly, we implement an existing deep learning model to compare two SPES analysis paradigms - namel
    
[^2]: 使用Transformer和RNN在经过训练的新损失函数下预测哺乳动物蛋白质中的O-GlcNAcylation位点

    Predicting O-GlcNAcylation Sites in Mammalian Proteins with Transformers and RNNs Trained with a New Loss Function

    [https://arxiv.org/abs/2402.17131](https://arxiv.org/abs/2402.17131)

    本研究提出了一种新的损失函数，称为加权焦点可微MCC，用于改善分类模型的性能，并在预测哺乳动物蛋白质中的O-GlcNAcylation位点方面取得了进展

    

    糖基化是一种蛋白质修饰，在功能和结构上起着多种重要作用。O-GlcNAcylation是糖基化的一种亚型，有潜力成为治疗的重要靶点，但在2023年之前尚未有可靠预测O-GlcNAcylation位点的方法；2021年的一篇评论正确指出已发表的模型不足，并且未能泛化。此外，许多模型已不再可用。2023年，一篇具有F$_1$分数36.17%和MCC分数34.57%的大型数据集上的显着更好的RNN模型被发表。本文首次试图通过Transformer编码器提高这些指标。尽管Transformer在该数据集上表现出色，但其性能仍不及先前发表的RNN。然后我们创建了一种新的损失函数，称为加权焦点可微MCC，以提高分类模型的性能。

    arXiv:2402.17131v1 Announce Type: new  Abstract: Glycosylation, a protein modification, has multiple essential functional and structural roles. O-GlcNAcylation, a subtype of glycosylation, has the potential to be an important target for therapeutics, but methods to reliably predict O-GlcNAcylation sites had not been available until 2023; a 2021 review correctly noted that published models were insufficient and failed to generalize. Moreover, many are no longer usable. In 2023, a considerably better RNN model with an F$_1$ score of 36.17% and an MCC of 34.57% on a large dataset was published. This article first sought to improve these metrics using transformer encoders. While transformers displayed high performance on this dataset, their performance was inferior to that of the previously published RNN. We then created a new loss function, which we call the weighted focal differentiable MCC, to improve the performance of classification models. RNN models trained with this new function di
    

