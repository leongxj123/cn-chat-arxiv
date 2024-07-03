# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Preserving Instructions for Aligning Large Language Models](https://arxiv.org/abs/2402.13659) | 提出使用合成指南替换真实指南以增强隐私保护，并通过私密微调生成器生成此类合成指南，并通过新颖的过滤算法使合成指南的分布与真实指南一致，展示了在大型语言模型对齐中的高效用性。 |
| [^2] | [A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models](https://arxiv.org/abs/2402.11469) | 本文研究了训练数据与模型鲁棒性之间的相关性，并提出通过提取不同特征来预测Transformer文本模型的对抗性稳健性的方法。 |
| [^3] | [Proceedings of the 2nd International Workshop on Adaptive Cyber Defense.](http://arxiv.org/abs/2308.09520) | 第二届自适应网络防御国际研讨会的目标是探索利用人工智能和机器学习作为自适应网络防御基础能力的研究，并通过填补AI和网络研究人员之间的差距来加速开发半自主网络防御系统。 |
| [^4] | [Selective Pre-training for Private Fine-tuning.](http://arxiv.org/abs/2305.13865) | 本文提出了一个通用框架，解决在保护隐私和满足内存和推理时间要求的情况下，在公共数据集上预训练一个固定大小的模型，并在私有数据集上进行微调以最大化对下游任务的性能。框架的关键是在公共数据集的子集上进行有选择性的预训练，使公共分布靠近私有分布。 |

# 详细

[^1]: 大型语言模型对齐的隐私保护指南

    Privacy-Preserving Instructions for Aligning Large Language Models

    [https://arxiv.org/abs/2402.13659](https://arxiv.org/abs/2402.13659)

    提出使用合成指南替换真实指南以增强隐私保护，并通过私密微调生成器生成此类合成指南，并通过新颖的过滤算法使合成指南的分布与真实指南一致，展示了在大型语言模型对齐中的高效用性。

    

    大型语言模型（LLM）应用的服务提供商在野外收集用户指南，并在进一步对齐LLM与用户意图中使用这些指南。这些潜在包含敏感信息的指南在流程中由人工工作者标注。这带来了新的隐私风险，而Typical Private Optimization没有解决这个问题。为此，我们提议使用合成指南替换数据标注和模型微调中的真实指南。通过使用私密微调生成器生成这些合成指南，可以确保形式差异隐私。在实现所需效用方面至关重要的是我们的新颖过滤算法，将合成指南的分布与实际指南的分布进行匹配。在有人反馈的受监督微调和强化学习中，我们的广泛实验表明，通过展示合成指南的最终集合的高效用性

    arXiv:2402.13659v1 Announce Type: cross  Abstract: Service providers of large language model (LLM) applications collect user instructions in the wild and use them in further aligning LLMs with users' intentions. These instructions, which potentially contain sensitive information, are annotated by human workers in the process. This poses a new privacy risk not addressed by the typical private optimization. To this end, we propose using synthetic instructions to replace real instructions in data annotation and model fine-tuning. Formal differential privacy is guaranteed by generating those synthetic instructions using privately fine-tuned generators. Crucial in achieving the desired utility is our novel filtering algorithm that matches the distribution of the synthetic instructions to that of the real ones. In both supervised fine-tuning and reinforcement learning from human feedback, our extensive experiments demonstrate the high utility of the final set of synthetic instructions by sho
    
[^2]: 在搜索训练数据与Transformer文本模型对抗性稳健性之间的相关性时的一个有趣案例

    A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models

    [https://arxiv.org/abs/2402.11469](https://arxiv.org/abs/2402.11469)

    本文研究了训练数据与模型鲁棒性之间的相关性，并提出通过提取不同特征来预测Transformer文本模型的对抗性稳健性的方法。

    

    现有研究表明，经过微调的文本Transformer模型可以实现最先进的预测性能，但也容易受到对抗文本扰动的影响。传统的对抗性评估通常在对模型进行微调之后才进行，忽略了训练数据。本文旨在证明训练数据和模型鲁棒性之间也存在着强关联。为此，我们提取了代表广泛输入微调语料库属性的13种不同特征，并用它们来预测经过微调的模型的对抗性稳健性。我们主要关注仅编码器的Transformer模型BERT和RoBERTa，并附加了BART、ELECTRA和GPT2的其他结果，为我们的论点提供多样的证据。首先，经验证明，(a)提取的特征可与轻量级分类器（如随机森林）一起有效地预测攻击成功率。

    arXiv:2402.11469v1 Announce Type: cross  Abstract: Existing works have shown that fine-tuned textual transformer models achieve state-of-the-art prediction performances but are also vulnerable to adversarial text perturbations. Traditional adversarial evaluation is often done \textit{only after} fine-tuning the models and ignoring the training data. In this paper, we want to prove that there is also a strong correlation between training data and model robustness. To this end, we extract 13 different features representing a wide range of input fine-tuning corpora properties and use them to predict the adversarial robustness of the fine-tuned models. Focusing mostly on encoder-only transformer models BERT and RoBERTa with additional results for BART, ELECTRA and GPT2, we provide diverse evidence to support our argument. First, empirical analyses show that (a) extracted features can be used with a lightweight classifier such as Random Forest to effectively predict the attack success rate 
    
[^3]: 第二届自适应网络防御国际研讨会论文集

    Proceedings of the 2nd International Workshop on Adaptive Cyber Defense. (arXiv:2308.09520v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2308.09520](http://arxiv.org/abs/2308.09520)

    第二届自适应网络防御国际研讨会的目标是探索利用人工智能和机器学习作为自适应网络防御基础能力的研究，并通过填补AI和网络研究人员之间的差距来加速开发半自主网络防御系统。

    

    第二届自适应网络防御国际研讨会在佛罗里达理工学院举行，该研讨会旨在分享利用人工智能（AI）和机器学习（ML）作为自适应网络防御基础能力的研究。当前的网络领域无法可靠有效地进行防御，必须广泛依赖人工专家。熟练的网络防御人员供应不足，往往无法及时应对网络威胁。借鉴AI和ML的最新进展，网络防御研究社区被激励着通过将AI和ML技术应用于网络环境中，开发新的动态可持续的防御措施。填补AI和网络研究人员与实践者之间的关键差距可以加速创建能够学习识别和应对网络攻击，或者发现和减轻弱点的半自主网络防御系统的努力。

    The 2nd International Workshop on Adaptive Cyber Defense was held at the Florida Institute of Technology, Florida. This workshop was organized to share research that explores unique applications of Artificial Intelligence (AI) and Machine Learning (ML) as foundational capabilities for the pursuit of adaptive cyber defense. The cyber domain cannot currently be reliably and effectively defended without extensive reliance on human experts. Skilled cyber defenders are in short supply and often cannot respond fast enough to cyber threats.  Building on recent advances in AI and ML the Cyber defense research community has been motivated to develop new dynamic and sustainable defenses through the adoption of AI and ML techniques to cyber settings. Bridging critical gaps between AI and Cyber researchers and practitioners can accelerate efforts to create semi-autonomous cyber defenses that can learn to recognize and respond to cyber attacks or discover and mitigate weaknesses in cooperation with
    
[^4]: 针对私有微调的有选择性预训练

    Selective Pre-training for Private Fine-tuning. (arXiv:2305.13865v1 [cs.LG])

    [http://arxiv.org/abs/2305.13865](http://arxiv.org/abs/2305.13865)

    本文提出了一个通用框架，解决在保护隐私和满足内存和推理时间要求的情况下，在公共数据集上预训练一个固定大小的模型，并在私有数据集上进行微调以最大化对下游任务的性能。框架的关键是在公共数据集的子集上进行有选择性的预训练，使公共分布靠近私有分布。

    

    假设我们想在电子邮件客户端或文字处理器中训练文本预测模型。这些模型必须保护用户数据的隐私，并遵守特定的固定大小，以满足内存和推理时间要求。我们介绍了一个通用框架来解决这个问题。具体来说，我们有一个公共数据集D_pub和一个对应于下游任务T的私有数据集D_priv。我们如何在D_pub上预训练一个固定大小的模型M，并在D_priv上微调它，使得M相对于T的性能最大化，并且M相对于D_priv具有差分隐私保护？我们展示了在D_pub的一个子集上预训练，将公共分布与私有分布靠近，是最大化M预训练后的迁移学习能力的关键因素，特别是在模型大小相对较小的情况下。除了性能改进外，我们的框架还提供了保护隐私的机制。

    Suppose we want to train text prediction models in email clients or word processors. The models must preserve the privacy of user data and adhere to a specific fixed size to meet memory and inference time requirements. We introduce a generic framework to solve this problem. Specifically, we are given a public dataset $D_\text{pub}$ and a private dataset $D_\text{priv}$ corresponding to a downstream task $T$. How should we pre-train a fixed-size model $M$ on $D_\text{pub}$ and fine-tune it on $D_\text{priv}$ such that performance of $M$ with respect to $T$ is maximized and $M$ satisfies differential privacy with respect to $D_\text{priv}$? We show that pre-training on a {\em subset} of dataset $D_\text{pub}$ that brings the public distribution closer to the private distribution is a crucial ingredient to maximize the transfer learning abilities of $M$ after pre-training, especially in the regimes where model sizes are relatively small. Besides performance improvements, our framework als
    

