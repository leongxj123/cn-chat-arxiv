# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Voice EHR: Introducing Multimodal Audio Data for Health](https://arxiv.org/abs/2404.01620) | 本报告引入了一种通过引导问题使用移动应用程序捕获健康数据的新的音频电子健康记录（voice EHR），可能包含复杂的健康生物标志物，从而弥补了单一模态临床数据的典型限制。 |
| [^2] | [Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees](https://arxiv.org/abs/2402.17106) | 该论文提出了一种针对数据集特性量身定制的近似公平性-准确性权衡曲线计算方法，能够有效减轻训练多个模型的计算负担并提供了严格的统计保证 |
| [^3] | [Focal Inferential Infusion Coupled with Tractable Density Discrimination for Implicit Hate Speech Detection.](http://arxiv.org/abs/2309.11896) | FiADD是一种新颖的焦点推理注入与易处理密度区分框架，通过将隐性仇恨言论的表面形式与暗示的形式更接近，同时增加不同类别标签之间的集群间距，显著改进了隐性仇恨分类任务的性能。 |

# 详细

[^1]: Voice EHR:引入多模式音频数据用于健康

    Voice EHR: Introducing Multimodal Audio Data for Health

    [https://arxiv.org/abs/2404.01620](https://arxiv.org/abs/2404.01620)

    本报告引入了一种通过引导问题使用移动应用程序捕获健康数据的新的音频电子健康记录（voice EHR），可能包含复杂的健康生物标志物，从而弥补了单一模态临床数据的典型限制。

    

    在音频数据上训练的大型AI模型可能具有快速分类患者的潜力，通过早期检测增强医疗决策，并可能通过早期检测改善结果。现有技术依赖于在高收入、英语国家使用昂贵记录设备的有限数据集，这种技术面临资源受限、高收入场所的部署挑战，音频数据可能具有深远影响。本报告介绍了一种新的数据类型和相应的收集系统，通过引导问题仅使用移动应用/网络应用程序捕获健康数据。该应用程序最终产生一个音频电子健康记录（voice EHR），它可能包含来自传统语音/呼吸特征、语音模式和具有语义意义的语言的复杂生物标志物，补偿单一模态临床数据的典型限制。本报告介绍了一个合作伙伴财团

    arXiv:2404.01620v1 Announce Type: cross  Abstract: Large AI models trained on audio data may have the potential to rapidly classify patients, enhancing medical decision-making and potentially improving outcomes through early detection. Existing technologies depend on limited datasets using expensive recording equipment in high-income, English-speaking countries. This challenges deployment in resource-constrained, high-volume settings where audio data may have a profound impact. This report introduces a novel data type and a corresponding collection system that captures health data through guided questions using only a mobile/web application. This application ultimately results in an audio electronic health record (voice EHR) which may contain complex biomarkers of health from conventional voice/respiratory features, speech patterns, and language with semantic meaning - compensating for the typical limitations of unimodal clinical datasets. This report introduces a consortium of partner
    
[^2]: 数据集公平性：在您的数据上实现具有效用保证的公平性

    Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees

    [https://arxiv.org/abs/2402.17106](https://arxiv.org/abs/2402.17106)

    该论文提出了一种针对数据集特性量身定制的近似公平性-准确性权衡曲线计算方法，能够有效减轻训练多个模型的计算负担并提供了严格的统计保证

    

    在机器学习公平性中，训练能够最小化不同敏感群体之间差异的模型通常会导致准确性下降，这种现象被称为公平性-准确性权衡。这种权衡的严重程度基本取决于数据集的特性，如数据集的不均衡或偏见。因此，在数据集之间使用统一的公平性要求仍然值得怀疑，并且往往会导致效用极低的模型。为了解决这个问题，我们提出了一种针对单个数据集量身定制的近似公平性-准确性权衡曲线的计算效率高的方法，该方法支持严格的统计保证。通过利用You-Only-Train-Once（YOTO）框架，我们的方法减轻了在逼近权衡曲线时需要训练多个模型的计算负担。此外，我们通过在该曲线周围引入置信区间来量化我们近似值的不确定性，

    arXiv:2402.17106v1 Announce Type: cross  Abstract: In machine learning fairness, training models which minimize disparity across different sensitive groups often leads to diminished accuracy, a phenomenon known as the fairness-accuracy trade-off. The severity of this trade-off fundamentally depends on dataset characteristics such as dataset imbalances or biases. Therefore using a uniform fairness requirement across datasets remains questionable and can often lead to models with substantially low utility. To address this, we present a computationally efficient approach to approximate the fairness-accuracy trade-off curve tailored to individual datasets, backed by rigorous statistical guarantees. By utilizing the You-Only-Train-Once (YOTO) framework, our approach mitigates the computational burden of having to train multiple models when approximating the trade-off curve. Moreover, we quantify the uncertainty in our approximation by introducing confidence intervals around this curve, offe
    
[^3]: 针对隐性仇恨言论的焦点推理注入与易处理密度区分

    Focal Inferential Infusion Coupled with Tractable Density Discrimination for Implicit Hate Speech Detection. (arXiv:2309.11896v1 [cs.CL])

    [http://arxiv.org/abs/2309.11896](http://arxiv.org/abs/2309.11896)

    FiADD是一种新颖的焦点推理注入与易处理密度区分框架，通过将隐性仇恨言论的表面形式与暗示的形式更接近，同时增加不同类别标签之间的集群间距，显著改进了隐性仇恨分类任务的性能。

    

    虽然预训练的大型语言模型（PLMs）在许多NLP任务上取得了最先进的成果，但它们缺乏对隐性仇恨言论微妙表达的理解。这样微妙而隐性的仇恨经常被错误地分类为非仇恨。通过增加外部的上下文或通过基于距离的度量强制标签分离，已经尝试过各种方法来增强（隐性）仇恨内容的检测。我们将这两种方法结合起来并引入了一种新颖的焦点推理适应密度区分框架（FiADD）。FiADD通过将隐性仇恨言论的表面形式与暗示的形式更接近，同时增加不同类别标签之间的集群间距，来增强PLM微调管道。我们在三个隐性仇恨数据集上测试了FiADD，并观察到在两类和三类仇恨分类任务中的显著改进。我们进一步对FiADD在三个其他任务上的泛化性进行了实验，即检测讽刺、讽刺和立场。

    Although pre-trained large language models (PLMs) have achieved state-of-the-art on many NLP tasks, they lack understanding of subtle expressions of implicit hate speech. Such nuanced and implicit hate is often misclassified as non-hate. Various attempts have been made to enhance the detection of (implicit) hate content by augmenting external context or enforcing label separation via distance-based metrics. We combine these two approaches and introduce FiADD, a novel Focused Inferential Adaptive Density Discrimination framework. FiADD enhances the PLM finetuning pipeline by bringing the surface form of an implicit hate speech closer to its implied form while increasing the inter-cluster distance among various class labels. We test FiADD on three implicit hate datasets and observe significant improvement in the two-way and three-way hate classification tasks. We further experiment on the generalizability of FiADD on three other tasks, namely detecting sarcasm, irony, and stance, in whic
    

