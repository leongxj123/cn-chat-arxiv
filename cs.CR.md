# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks.](http://arxiv.org/abs/2401.04929) | 本文介绍了一种基于学习的难度校准的成员推理攻击方法，旨在显著提高低FPR下的TPR，以验证训练模型是否保护隐私。 |
| [^2] | [AUTOLYCUS: Exploiting Explainable AI (XAI) for Model Extraction Attacks against White-Box Models.](http://arxiv.org/abs/2302.02162) | 本文探究了可解释人工智能（XAI）工具对机器学习模型提取攻击的风险，并提出了一种模型提取攻击方法AUTOLYCUS。 |
| [^3] | [Does CLIP Know My Face?.](http://arxiv.org/abs/2209.07341) | 本文提出了一种新方法IDIA来评估视觉语言模型的隐私，大规模实验表明使用于训练的个人可以被非常高的准确率识别出来，表明需要更好地解决视觉语言模型中的隐私问题。 |

# 详细

[^1]: 基于学习的难度校准提升成员推理攻击的能力

    Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks. (arXiv:2401.04929v1 [cs.CR])

    [http://arxiv.org/abs/2401.04929](http://arxiv.org/abs/2401.04929)

    本文介绍了一种基于学习的难度校准的成员推理攻击方法，旨在显著提高低FPR下的TPR，以验证训练模型是否保护隐私。

    

    机器学习模型，特别是深度神经网络，目前是各种应用的重要组成部分，从医疗保健到金融。然而，使用敏感数据来训练这些模型引发了对隐私和安全的担忧。一种验证训练模型是否保护隐私的方法是成员推理攻击（MIA），它允许对手确定特定数据点是否是模型的训练数据集的一部分。虽然已经在文献中提出了一系列的MIA，但只有少数能够在低假阳性率（FPR）区域（0.01%~1%）实现较高的真阳性率（TPR）。这是实际应用于实际场景中的MIA必须考虑的关键因素。在本文中，我们提出了一种新颖的MIA方法，旨在显著提高低FPR的TPR。我们的方法名为基于学习的难度校准（LDC-MIA），通过使用神经网络分类器将数据记录以其难度级别进行表征。

    Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network clas
    
[^2]: AUTOLYCUS: 利用可解释人工智能（XAI）对白盒模型进行模型提取攻击

    AUTOLYCUS: Exploiting Explainable AI (XAI) for Model Extraction Attacks against White-Box Models. (arXiv:2302.02162v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02162](http://arxiv.org/abs/2302.02162)

    本文探究了可解释人工智能（XAI）工具对机器学习模型提取攻击的风险，并提出了一种模型提取攻击方法AUTOLYCUS。

    

    可解释的人工智能（XAI）涵盖了一系列旨在阐明AI模型决策过程的技术和程序。虽然XAI对于理解AI模型的推理过程很有价值，但用于这种揭示的数据会带来潜在的安全和隐私漏洞。现有文献已经确定了针对机器学习模型的隐私风险，包括成员推论、模型反演和模型提取攻击。根据涉及的设置和各方，这些攻击可能针对模型本身或用于创建模型的训练数据。我们认为提供XAI的工具特别会增加模型提取攻击的风险，这可能是一个重要问题，当AI模型的所有者仅愿提供黑盒访问而不与其他方共享模型参数和结构时。为了探究这种隐私风险，我们提出了AUTOLYCUS，一种模型提取攻击方法。

    Explainable Artificial Intelligence (XAI) encompasses a range of techniques and procedures aimed at elucidating the decision-making processes of AI models. While XAI is valuable in understanding the reasoning behind AI models, the data used for such revelations poses potential security and privacy vulnerabilities. Existing literature has identified privacy risks targeting machine learning models, including membership inference, model inversion, and model extraction attacks. Depending on the settings and parties involved, such attacks may target either the model itself or the training data used to create the model.  We have identified that tools providing XAI can particularly increase the vulnerability of model extraction attacks, which can be a significant issue when the owner of an AI model prefers to provide only black-box access rather than sharing the model parameters and architecture with other parties. To explore this privacy risk, we propose AUTOLYCUS, a model extraction attack 
    
[^3]: CLIP是否知道我的脸？

    Does CLIP Know My Face?. (arXiv:2209.07341v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.07341](http://arxiv.org/abs/2209.07341)

    本文提出了一种新方法IDIA来评估视觉语言模型的隐私，大规模实验表明使用于训练的个人可以被非常高的准确率识别出来，表明需要更好地解决视觉语言模型中的隐私问题。

    

    随着深度学习在各个应用中的普及，保护训练数据的隐私问题已经成为一个关键的研究领域。以前的研究主要关注单模型的隐私风险，我们提出了一种新的方法来评估多模型的隐私，特别是像CLIP这样的视觉语言模型。所提出的身份推断攻击(IDIA)通过用同一人的图片向模型查询，从而揭示该个人是否被包含在训练数据中。让模型从各种可能的文本标签中选择，模型会透露是否识别该人物，从而表明其被用于训练。我们在CLIP上进行的大规模实验表明，使用于训练的个人可以被非常高的准确率识别出来。我们确认该模型已经学会将名称与描绘的个人相关联，这意味着敏感信息存在于其中，可以被对手提取。我们的结果凸显了需要在视觉语言模型中更好地解决隐私问题。

    With the rise of deep learning in various applications, privacy concerns around the protection of training data has become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for 
    

