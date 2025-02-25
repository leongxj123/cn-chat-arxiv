# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [One-Shot Domain Incremental Learning](https://arxiv.org/abs/2403.16707) | 提出了一种处理单次领域增量学习中批归一化层统计数据困难的技术，并展示了其有效性。 |
| [^2] | [Tur[k]ingBench: A Challenge Benchmark for Web Agents](https://arxiv.org/abs/2403.11905) | Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。 |
| [^3] | [CIC: A framework for Culturally-aware Image Captioning](https://arxiv.org/abs/2402.05374) | CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。 |
| [^4] | [Your Diffusion Model is Secretly a Certifiably Robust Classifier](https://arxiv.org/abs/2402.02316) | 这项研究提出了一种新的扩散分类器家族，称为噪声扩散分类器（NDCs），其具有最新的可证明的鲁棒性。通过将扩散分类器推广到分类高斯受损数据，并将其与随机平滑技术相结合，构建了具有非常量Lipschitzness的平滑分类器。这些NDCs显示出卓越的认证鲁棒性。 |
| [^5] | [SoK: Facial Deepfake Detectors.](http://arxiv.org/abs/2401.04364) | 本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。 |
| [^6] | [MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems.](http://arxiv.org/abs/2309.08421) | 本研究开发了一种新颖的机器学习框架MIML，该框架通过将无标记细胞图像与生物力学属性数据相结合，实现了高精度细胞分类。该方法利用了形态信息，将细胞属性理解得更全面，相较于仅考虑单一数据类型的模型，实现了98.3％的分类精度。该方法已在白细胞和肿瘤细胞分类中得到证明，并具有更广泛的应用潜力。 |

# 详细

[^1]: 单次领域增量学习

    One-Shot Domain Incremental Learning

    [https://arxiv.org/abs/2403.16707](https://arxiv.org/abs/2403.16707)

    提出了一种处理单次领域增量学习中批归一化层统计数据困难的技术，并展示了其有效性。

    

    在以前关于用于分类的深度神经网络模型的研究中已经讨论了领域增量学习（DIL）。在DIL中，我们假设随着时间的推移观察新领域上的样本。模型必须对所有领域上的输入进行分类。然而，在实践中，我们可能会遇到这样一种情况，即我们需要在新领域的样本仅间歇性地被观察的约束下执行DIL。因此，在本研究中，我们考虑了一个极端情况，即我们只有一份来自新领域的样本，我们称之为单次DIL。我们首先经验性地表明现有的DIL方法在单次DIL中表现不佳。通过各种调查，我们分析了这种失败的原因。根据我们的分析，我们明确了单次DIL的困难是由批归一化层中的统计数据引起的。因此，我们提出了一种关于这些统计数据的技术，并展示了我们技术的有效性。

    arXiv:2403.16707v1 Announce Type: cross  Abstract: Domain incremental learning (DIL) has been discussed in previous studies on deep neural network models for classification. In DIL, we assume that samples on new domains are observed over time. The models must classify inputs on all domains. In practice, however, we may encounter a situation where we need to perform DIL under the constraint that the samples on the new domain are observed only infrequently. Therefore, in this study, we consider the extreme case where we have only one sample from the new domain, which we call one-shot DIL. We first empirically show that existing DIL methods do not work well in one-shot DIL. We have analyzed the reason for this failure through various investigations. According to our analysis, we clarify that the difficulty of one-shot DIL is caused by the statistics in the batch normalization layers. Therefore, we propose a technique regarding these statistics and demonstrate the effectiveness of our tech
    
[^2]: Tur[k]ingBench：用于网络代理的挑战基准测试

    Tur[k]ingBench: A Challenge Benchmark for Web Agents

    [https://arxiv.org/abs/2403.11905](https://arxiv.org/abs/2403.11905)

    Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。

    

    最近的聊天机器人展示了在原始文本形式下理解和交流的令人印象深刻的能力。然而，世界上不仅仅是原始文本。例如，人们在网页上花费大量时间，在这些网页上，文本与其他形式交织在一起，并以各种复杂互动的形式完成任务。最先进的多模型是否能够推广到这种复杂的领域呢？为了回答这个问题，我们介绍了TurkingBench，一个由包含多模态背景的文本说明制定的任务基准。与现有的使用人工合成的网页的工作不同，这里我们使用最初设计用于各种注释目的的自然HTML页面。每个任务的HTML说明也被实例化为各种值（从众包任务获得）以形成任务的新实例。这个基准包含32.2K个实例。

    arXiv:2403.11905v1 Announce Type: new  Abstract: Recent chatbots have demonstrated impressive ability to understand and communicate in raw-text form. However, there is more to the world than raw text. For example, humans spend long hours of their time on web pages, where text is intertwined with other modalities and tasks are accomplished in the form of various complex interactions. Can state-of-the-art multi-modal models generalize to such complex domains?   To address this question, we introduce TurkingBench, a benchmark of tasks formulated as web pages containing textual instructions with multi-modal context. Unlike existing work which employs artificially synthesized web pages, here we use natural HTML pages that were originally designed for crowdsourcing workers for various annotation purposes. The HTML instructions of each task are also instantiated with various values (obtained from the crowdsourcing tasks) to form new instances of the task. This benchmark contains 32.2K instanc
    
[^3]: CIC：一种面向文化感知图像字幕的框架

    CIC: A framework for Culturally-aware Image Captioning

    [https://arxiv.org/abs/2402.05374](https://arxiv.org/abs/2402.05374)

    CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。

    

    图像字幕通过使用视觉-语言预训练模型（VLPs）如BLIP从图像生成描述性句子，这种方法已经取得了很大的改进。然而，当前的方法缺乏对图像中所描绘的文化元素（例如亚洲文化群体的传统服装）生成详细描述性字幕的能力。在本文中，我们提出了一种新的框架，\textbf{面向文化感知图像字幕（CIC）}，该框架能够从代表不同文化的图像中生成字幕并描述文化元素。受到将视觉模态和大型语言模型（LLMs）通过适当的提示进行组合的方法的启发，我们的框架（1）根据图像中的文化类别生成问题，（2）利用生成的问题从视觉问答（VQA）中提取文化视觉元素，（3）使用带有提示的LLMs生成文化感知字幕。我们在4个不同大学的45名参与者上进行了人工评估。

    Image Captioning generates descriptive sentences from images using Vision-Language Pre-trained models (VLPs) such as BLIP, which has improved greatly. However, current methods lack the generation of detailed descriptive captions for the cultural elements depicted in the images, such as the traditional clothing worn by people from Asian cultural groups. In this paper, we propose a new framework, \textbf{Culturally-aware Image Captioning (CIC)}, that generates captions and describes cultural elements extracted from cultural visual elements in images representing cultures. Inspired by methods combining visual modality and Large Language Models (LLMs) through appropriate prompts, our framework (1) generates questions based on cultural categories from images, (2) extracts cultural visual elements from Visual Question Answering (VQA) using generated questions, and (3) generates culturally-aware captions using LLMs with the prompts. Our human evaluation conducted on 45 participants from 4 dif
    
[^4]: 你的扩散模型实际上是一个可证明鲁棒的分类器

    Your Diffusion Model is Secretly a Certifiably Robust Classifier

    [https://arxiv.org/abs/2402.02316](https://arxiv.org/abs/2402.02316)

    这项研究提出了一种新的扩散分类器家族，称为噪声扩散分类器（NDCs），其具有最新的可证明的鲁棒性。通过将扩散分类器推广到分类高斯受损数据，并将其与随机平滑技术相结合，构建了具有非常量Lipschitzness的平滑分类器。这些NDCs显示出卓越的认证鲁棒性。

    

    近期，扩散模型被作为鲁棒分类的生成器分类器所采用。然而，对于扩散分类器鲁棒性的综合理论理解仍然缺乏，这让我们怀疑它们是否会容易受到未来更强攻击的影响。在本研究中，我们提出了一种新的扩散分类器家族，命名为噪声扩散分类器（NDCs），其具有最新的可证明的鲁棒性。具体来说，我们通过推导这些分布的证据下界（ELBOs），利用ELBO近似似然度量，并使用贝叶斯定理计算分类概率，将扩散分类器推广到分类高斯受损数据。我们将这些推广的扩散分类器与随机平滑技术相结合，构建具有非常量Lipschitzness的平滑分类器。实验结果表明我们提出的NDCs在鲁棒性方面具有卓越的认证能力。值得注意的是，我们是第一个达到80%的...

    Diffusion models are recently employed as generative classifiers for robust classification. However, a comprehensive theoretical understanding of the robustness of diffusion classifiers is still lacking, leading us to question whether they will be vulnerable to future stronger attacks. In this study, we propose a new family of diffusion classifiers, named Noised Diffusion Classifiers~(NDCs), that possess state-of-the-art certified robustness. Specifically, we generalize the diffusion classifiers to classify Gaussian-corrupted data by deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. We integrate these generalized diffusion classifiers with randomized smoothing to construct smoothed classifiers possessing non-constant Lipschitzness. Experimental results demonstrate the superior certified robustness of our proposed NDCs. Notably, we are the first to achieve 80\%
    
[^5]: SoK：面部深度伪造检测器

    SoK: Facial Deepfake Detectors. (arXiv:2401.04364v1 [cs.CV])

    [http://arxiv.org/abs/2401.04364](http://arxiv.org/abs/2401.04364)

    本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。

    

    深度伪造技术迅速成为对社会构成深远和严重威胁的原因之一，主要由于其易于制作和传播。这种情况加速了深度伪造检测技术的发展。然而，许多现有的检测器在验证时 heavily 依赖实验室生成的数据集，这可能无法有效地让它们应对新颖、新兴和实际的深度伪造技术。本文对最新的深度伪造检测器进行广泛全面的回顾和分析，根据几个关键标准对它们进行评估。这些标准将这些检测器分为 4 个高级组别和 13 个细粒度子组别，都遵循一个统一的标准概念框架。这种分类和框架提供了对影响检测器功效的因素的深入和实用的见解。我们对 16 个主要的检测器在各种标准的攻击场景中的普适性进行评估，包括黑盒攻击场景。

    Deepfakes have rapidly emerged as a profound and serious threat to society, primarily due to their ease of creation and dissemination. This situation has triggered an accelerated development of deepfake detection technologies. However, many existing detectors rely heavily on lab-generated datasets for validation, which may not effectively prepare them for novel, emerging, and real-world deepfake techniques. In this paper, we conduct an extensive and comprehensive review and analysis of the latest state-of-the-art deepfake detectors, evaluating them against several critical criteria. These criteria facilitate the categorization of these detectors into 4 high-level groups and 13 fine-grained sub-groups, all aligned with a unified standard conceptual framework. This classification and framework offer deep and practical insights into the factors that affect detector efficacy. We assess the generalizability of 16 leading detectors across various standard attack scenarios, including black-bo
    
[^6]: MIML: 通过微流控系统内的机械特性对高精度细胞分类进行多重图像机器学习

    MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems. (arXiv:2309.08421v1 [eess.IV])

    [http://arxiv.org/abs/2309.08421](http://arxiv.org/abs/2309.08421)

    本研究开发了一种新颖的机器学习框架MIML，该框架通过将无标记细胞图像与生物力学属性数据相结合，实现了高精度细胞分类。该方法利用了形态信息，将细胞属性理解得更全面，相较于仅考虑单一数据类型的模型，实现了98.3％的分类精度。该方法已在白细胞和肿瘤细胞分类中得到证明，并具有更广泛的应用潜力。

    

    无标记细胞分类有助于为进一步使用或检查提供原始细胞，然而现有技术在特异性和速度方面往往不足。在本研究中，我们通过开发一种新颖的机器学习框架MIML来解决这些局限性。该架构将无标记细胞图像与生物力学属性数据相结合，利用每个细胞固有的广阔且常常被低估的形态信息。通过整合这两种类型的数据，我们的模型提供了对细胞属性更全面的理解，利用了传统机器学习模型中通常被丢弃的形态信息。这种方法使细胞分类精度达到了惊人的98.3％，大大优于仅考虑单一数据类型的模型。MIML已被证明在白细胞和肿瘤细胞分类中有效，并具有更广泛的应用潜力。

    Label-free cell classification is advantageous for supplying pristine cells for further use or examination, yet existing techniques frequently fall short in terms of specificity and speed. In this study, we address these limitations through the development of a novel machine learning framework, Multiplex Image Machine Learning (MIML). This architecture uniquely combines label-free cell images with biomechanical property data, harnessing the vast, often underutilized morphological information intrinsic to each cell. By integrating both types of data, our model offers a more holistic understanding of the cellular properties, utilizing morphological information typically discarded in traditional machine learning models. This approach has led to a remarkable 98.3\% accuracy in cell classification, a substantial improvement over models that only consider a single data type. MIML has been proven effective in classifying white blood cells and tumor cells, with potential for broader applicatio
    

