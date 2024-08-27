# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning](https://arxiv.org/abs/2403.08525) | 提出一种基于自适应变点检测和主动学习的音频录制分割方法，通过预测模型和变点检测逐步生成高质量的强标签。 |
| [^2] | [SpeechDPR: End-to-End Spoken Passage Retrieval for Open-Domain Spoken Question Answering.](http://arxiv.org/abs/2401.13463) | SpeechDPR是第一个用于开放领域口语问答的端到端框架，能够从口语存档中检索可能包含答案的段落。通过融合无监督ASR和文本密集检索器的知识，SpeechDPR能够获得较好的性能，并且在UASR性能较差时表现更加鲁棒。 |
| [^3] | [Self-Supervised Disentangled Representation Learning for Robust Target Speech Extraction.](http://arxiv.org/abs/2312.10305) | 该论文提出了一种自监督分解表示学习方法，通过逐步分离说话人身份信息和其他无关因素，解决了目标语音提取任务中存在的说话人混叠问题，并使用分解的说话人身份信息来指导语音提取网络。 |

# 详细

[^1]: 从弱到强：使用自适应变点检测和主动学习进行声音事件标签

    From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning

    [https://arxiv.org/abs/2403.08525](https://arxiv.org/abs/2403.08525)

    提出一种基于自适应变点检测和主动学习的音频录制分割方法，通过预测模型和变点检测逐步生成高质量的强标签。

    

    在这项工作中，我们提出了一种基于自适应变点检测（A-CPD）的音频录制分割方法，用于机器引导的音频录制段的弱标签注释。目标是最大化关于目标声音时间激活的信息获取量。对于每个未标记的音频录制，我们使用预测模型来推导概率曲线，用于指导注释。预测模型最初在可用的带标注声音事件数据上进行预训练，这些数据的类与未标记数据集中的类不相交。然后，预测模型逐渐适应注释者在主动学习循环中提供的注释。用于引导弱标签注释者走向强标签的查询是使用这些概率上的变点检测导出的。我们展示，即使在有限的注释预算下，也可以获得高质量的强标签，并展示了优势。

    arXiv:2403.08525v1 Announce Type: cross  Abstract: In this work we propose an audio recording segmentation method based on an adaptive change point detection (A-CPD) for machine guided weak label annotation of audio recording segments. The goal is to maximize the amount of information gained about the temporal activation's of the target sounds. For each unlabeled audio recording, we use a prediction model to derive a probability curve used to guide annotation. The prediction model is initially pre-trained on available annotated sound event data with classes that are disjoint from the classes in the unlabeled dataset. The prediction model then gradually adapts to the annotations provided by the annotator in an active learning loop. The queries used to guide the weak label annotator towards strong labels are derived using change point detection on these probabilities. We show that it is possible to derive strong labels of high quality even with a limited annotation budget, and show favor
    
[^2]: SpeechDPR: 开放领域口语问答的端到端口语段落检索

    SpeechDPR: End-to-End Spoken Passage Retrieval for Open-Domain Spoken Question Answering. (arXiv:2401.13463v1 [cs.CL])

    [http://arxiv.org/abs/2401.13463](http://arxiv.org/abs/2401.13463)

    SpeechDPR是第一个用于开放领域口语问答的端到端框架，能够从口语存档中检索可能包含答案的段落。通过融合无监督ASR和文本密集检索器的知识，SpeechDPR能够获得较好的性能，并且在UASR性能较差时表现更加鲁棒。

    

    口语问答(SQA)是机器通过在给定口语段落中找到答案范围来回答用户问题的关键。过去的SQA方法没有使用ASR，以避免识别错误和词汇外问题。然而，实际的开放领域SQA(openSQA)问题中，机器需要首先从口语存档中检索可能包含答案的段落。本文提出了第一个已知的用于openSQA问题检索组件的端到端框架SpeechDPR。SpeechDPR通过从无监督ASR(UASR)和文本密集检索器(TDR)的级联模型中提炼知识，学习句子级语义表示。不需要手动转录的语音数据。初步实验表明，与级联的UASR和TDR模型相比，性能相当，并且在UASR性能较差时显著提高，验证了这种方法更加鲁棒。

    Spoken Question Answering (SQA) is essential for machines to reply to user's question by finding the answer span within a given spoken passage. SQA has been previously achieved without ASR to avoid recognition errors and Out-of-Vocabulary (OOV) problems. However, the real-world problem of Open-domain SQA (openSQA), in which the machine needs to first retrieve passages that possibly contain the answer from a spoken archive in addition, was never considered. This paper proposes the first known end-to-end framework, Speech Dense Passage Retriever (SpeechDPR), for the retrieval component of the openSQA problem. SpeechDPR learns a sentence-level semantic representation by distilling knowledge from the cascading model of unsupervised ASR (UASR) and text dense retriever (TDR). No manually transcribed speech data is needed. Initial experiments showed performance comparable to the cascading model of UASR and TDR, and significantly better when UASR was poor, verifying this approach is more robus
    
[^3]: 自监督分解表示学习用于鲁棒目标语音提取

    Self-Supervised Disentangled Representation Learning for Robust Target Speech Extraction. (arXiv:2312.10305v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2312.10305](http://arxiv.org/abs/2312.10305)

    该论文提出了一种自监督分解表示学习方法，通过逐步分离说话人身份信息和其他无关因素，解决了目标语音提取任务中存在的说话人混叠问题，并使用分解的说话人身份信息来指导语音提取网络。

    

    语音信号本质上是复杂的，因为它包含全局声学特征和局部语义信息。然而，在目标语音提取任务中，参考语音中与说话人身份无关的全局和局部语义信息可能导致在语音提取网络中出现说话人混叠问题。为了克服这个挑战，我们提出了一种自监督分解表示学习方法。我们的方法通过一个两阶段过程来解决这个问题，利用参考语音编码网络和全局信息分解网络逐渐分解说话人身份信息和其他不相关因素。我们专门使用分解的说话人身份信息来指导语音提取网络。此外，我们引入自适应调制Transformer来确保混合信号的声学表示不受说话人嵌入的影响。

    Speech signals are inherently complex as they encompass both global acoustic characteristics and local semantic information. However, in the task of target speech extraction, certain elements of global and local semantic information in the reference speech, which are irrelevant to speaker identity, can lead to speaker confusion within the speech extraction network. To overcome this challenge, we propose a self-supervised disentangled representation learning method. Our approach tackles this issue through a two-phase process, utilizing a reference speech encoding network and a global information disentanglement network to gradually disentangle the speaker identity information from other irrelevant factors. We exclusively employ the disentangled speaker identity information to guide the speech extraction network. Moreover, we introduce the adaptive modulation Transformer to ensure that the acoustic representation of the mixed signal remains undisturbed by the speaker embeddings. This com
    

