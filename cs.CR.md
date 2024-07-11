# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution](https://arxiv.org/abs/2305.17000) | DistriBlock提出了一种能够识别对抗性音频样本的有效检测策略，通过利用输出分布的特征，包括中位数、最大值和最小值、熵以及与后续时间步骤的分布之间的散度，应用二元分类器进行预测。这项研究证明了DistriBlock在识别对抗性音频样本方面的有效性。 |
| [^2] | [The Ethics of Interaction: Mitigating Security Threats in LLMs.](http://arxiv.org/abs/2401.12273) | 本研究全面探讨了与语言学习模型（LLMs）面临的安全威胁相关的伦理挑战。分析了五种主要威胁的伦理后果，并强调了确保这些系统在伦理规范范围内运作的紧迫性。 |
| [^3] | [Phishing Website Detection through Multi-Model Analysis of HTML Content.](http://arxiv.org/abs/2401.04820) | 本研究提出了一种基于HTML内容的高级检测模型，集成了多层感知器和预训练的自然语言处理模型，通过新颖的融合方法检测网络钓鱼网站。同时，我们还创造了一个最新的数据集来支持这项研究。 |
| [^4] | [SecureReg: A Combined Framework for Proactively Exposing Malicious Domain Name Registrations.](http://arxiv.org/abs/2401.03196) | SecureReg是一个结合了自然语言处理和多层感知器模型的方法，用于在域名注册过程中主动暴露恶意域名注册，提供了早期威胁检测的解决方案，显著减少了漏洞窗口，并为主动预防性操作做出了贡献。 |
| [^5] | [LLM-assisted Generation of Hardware Assertions.](http://arxiv.org/abs/2306.14027) | 本论文研究使用LLMs来生成硬件的安全断言。通过使用自然语言提示生成SystemVerilog断言来替代编写具有挑战的安全断言。 |
| [^6] | [Balancing Privacy and Security in Federated Learning with FedGT: A Group Testing Framework.](http://arxiv.org/abs/2305.05506) | 该论文提出了FedGT框架，通过群体测试的方法在联邦学习中识别并删除恶意客户，从而平衡了隐私和安全，保护数据隐私并提高了识别恶意客户的能力。 |

# 详细

[^1]: DistriBlock: 通过利用输出分布的特征识别对抗性音频样本

    DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution

    [https://arxiv.org/abs/2305.17000](https://arxiv.org/abs/2305.17000)

    DistriBlock提出了一种能够识别对抗性音频样本的有效检测策略，通过利用输出分布的特征，包括中位数、最大值和最小值、熵以及与后续时间步骤的分布之间的散度，应用二元分类器进行预测。这项研究证明了DistriBlock在识别对抗性音频样本方面的有效性。

    

    对抗性攻击可能误导自动语音识别（ASR）系统，使其预测任意目标文本，从而构成明显的安全威胁。为了防止这种攻击，我们提出了DistriBlock，一种适用于任何ASR系统的高效检测策略，该系统在每个时间步骤上预测输出标记的概率分布。我们对该分布的一组特征进行测量：输出概率的中位数、最大值和最小值，分布的熵，以及与后续时间步骤的分布之间的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性和对抗性数据观察到的特征，我们应用二元分类器，包括简单的基于阈值的分类、这种分类器的集合以及神经网络。通过对不同最先进的ASR系统和语言数据集进行广泛分析，我们证明了DistriBlock在识别对抗性音频样本方面的有效性。

    arXiv:2305.17000v2 Announce Type: replace-cross  Abstract: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, 
    
[^2]: 交互的伦理问题：缓解LLMs中的安全威胁

    The Ethics of Interaction: Mitigating Security Threats in LLMs. (arXiv:2401.12273v1 [cs.CR])

    [http://arxiv.org/abs/2401.12273](http://arxiv.org/abs/2401.12273)

    本研究全面探讨了与语言学习模型（LLMs）面临的安全威胁相关的伦理挑战。分析了五种主要威胁的伦理后果，并强调了确保这些系统在伦理规范范围内运作的紧迫性。

    

    本文全面探讨了与语言学习模型（LLMs）面临的安全威胁相关的伦理挑战。这些复杂的数字存储库日益融入到我们的日常生活中，因此成为攻击的主要目标，可能危及其训练数据和数据源的机密性。本文深入研究了这些安全威胁对社会和个人隐私的微妙伦理影响。我们对五个主要威胁进行了详细分析：提示注入、越狱、个人可识别信息（PII）曝露、性别显露内容和基于仇恨的内容。我们不仅仅进行了识别，还评估了它们的关键伦理后果以及对强化防御策略的紧迫性。对LLMs的不断依赖凸显了确保这些系统在伦理规范范围内运作的重要性，特别是由于它们的滥用可能导致重大社会和个人伤害。我们提出了将这些系统概念化的要求。

    This paper comprehensively explores the ethical challenges arising from security threats to Language Learning Models (LLMs). These intricate digital repositories are increasingly integrated into our daily lives, making them prime targets for attacks that can compromise their training data and the confidentiality of their data sources. The paper delves into the nuanced ethical repercussions of such security threats on society and individual privacy. We scrutinize five major threats: prompt injection, jailbreaking, Personal Identifiable Information (PII) exposure, sexually explicit content, and hate based content, going beyond mere identification to assess their critical ethical consequences and the urgency they create for robust defensive strategies. The escalating reliance on LLMs underscores the crucial need for ensuring these systems operate within the bounds of ethical norms, particularly as their misuse can lead to significant societal and individual harm. We propose conceptualizin
    
[^3]: 通过HTML内容的多模型分析检测网络钓鱼网站

    Phishing Website Detection through Multi-Model Analysis of HTML Content. (arXiv:2401.04820v1 [cs.CR])

    [http://arxiv.org/abs/2401.04820](http://arxiv.org/abs/2401.04820)

    本研究提出了一种基于HTML内容的高级检测模型，集成了多层感知器和预训练的自然语言处理模型，通过新颖的融合方法检测网络钓鱼网站。同时，我们还创造了一个最新的数据集来支持这项研究。

    

    随着互联网的兴起，我们的通信和工作方式发生了巨大的变化。虽然它为我们带来了新的机会，但也增加了网络威胁。其中一种常见且严重的威胁是网络钓鱼，黑客使用欺骗性方法窃取敏感信息。本研究通过引入一种基于HTML内容的先进检测模型，针对网络钓鱼问题进行了探讨。我们提出的方法集成了用于结构化表格数据的专门的多层感知器(MLP)模型和两个预训练的自然语言处理(NLP)模型，以分析页面标题和内容等文本特征。通过一种新颖的融合过程，这些模型生成的嵌入向量被和谐地组合在一起，并输入到线性分类器中。鉴于目前缺乏全面的网络钓鱼研究数据集，我们的贡献还包括创建一个最新的数据集

    The way we communicate and work has changed significantly with the rise of the Internet. While it has opened up new opportunities, it has also brought about an increase in cyber threats. One common and serious threat is phishing, where cybercriminals employ deceptive methods to steal sensitive information.This study addresses the pressing issue of phishing by introducing an advanced detection model that meticulously focuses on HTML content. Our proposed approach integrates a specialized Multi-Layer Perceptron (MLP) model for structured tabular data and two pretrained Natural Language Processing (NLP) models for analyzing textual features such as page titles and content. The embeddings from these models are harmoniously combined through a novel fusion process. The resulting fused embeddings are then input into a linear classifier. Recognizing the scarcity of recent datasets for comprehensive phishing research, our contribution extends to the creation of an up-to-date dataset, which we o
    
[^4]: SecureReg:一个结合方法用于主动暴露恶意域名注册

    SecureReg: A Combined Framework for Proactively Exposing Malicious Domain Name Registrations. (arXiv:2401.03196v1 [cs.CR])

    [http://arxiv.org/abs/2401.03196](http://arxiv.org/abs/2401.03196)

    SecureReg是一个结合了自然语言处理和多层感知器模型的方法，用于在域名注册过程中主动暴露恶意域名注册，提供了早期威胁检测的解决方案，显著减少了漏洞窗口，并为主动预防性操作做出了贡献。

    

    随着网络安全威胁不断增加，不法分子每天注册数千个新域名进行垃圾邮件、网络钓鱼和驱动下载等互联网攻击，强调了创新检测方法的需求。本文介绍了一种先进的方法，用于在注册过程开始时识别可疑域名。附带的数据流程通过比较新域名与注册域名产生关键特征，强调了关键相似度得分。利用自然语言处理（NLP）技术的新颖组合，包括预训练的Canine模型和多层感知器（MLP）模型，我们的系统分析语义和数值属性，为早期威胁检测提供了强大的解决方案。该综合方法显著减少了漏洞窗口，加强了对潜在威胁的防御。研究结果证明了该综合方法的有效性，并为开发主动预防性操作的努力做出了贡献。

    Rising cyber threats, with miscreants registering thousands of new domains daily for Internet-scale attacks like spam, phishing, and drive-by downloads, emphasize the need for innovative detection methods. This paper introduces a cutting-edge approach for identifying suspicious domains at the onset of the registration process. The accompanying data pipeline generates crucial features by comparing new domains to registered domains,emphasizing the crucial similarity score. Leveraging a novel combination of Natural Language Processing (NLP) techniques, including a pretrained Canine model, and Multilayer Perceptron (MLP) models, our system analyzes semantic and numerical attributes, providing a robust solution for early threat detection. This integrated approach significantly reduces the window of vulnerability, fortifying defenses against potential threats. The findings demonstrate the effectiveness of the integrated approach and contribute to the ongoing efforts in developing proactive s
    
[^5]: 基于LLM的硬件断言生成辅助

    LLM-assisted Generation of Hardware Assertions. (arXiv:2306.14027v1 [cs.CR])

    [http://arxiv.org/abs/2306.14027](http://arxiv.org/abs/2306.14027)

    本论文研究使用LLMs来生成硬件的安全断言。通过使用自然语言提示生成SystemVerilog断言来替代编写具有挑战的安全断言。

    

    计算机系统的安全性通常依赖于硬件的安全性。硬件漏洞对系统有严重影响，因此需要技术支持安全验证活动。断言验证是一种流行的验证技术，它涉及在一组断言中捕捉设计意图，这些断言可用于形式验证或基于测试的检查。然而，编写以安全为中心的断言是一项具有挑战性的任务。在本研究中，我们探讨使用新型大型语言模型（LLMs）进行硬件断言生成的代码生成技术，其中主要使用自然语言提示（例如在断言文件中看到的代码注释）生成SystemVerilog断言。我们关注一种流行的LLM，并对其在给定不同详细级别的提示的情况下编写断言的能力进行了表征。我们设计了一个评估框架，生成各种LLM辅助下产生的系统断言形式进行评估。

    The security of computer systems typically relies on a hardware root of trust. As vulnerabilities in hardware can have severe implications on a system, there is a need for techniques to support security verification activities. Assertion-based verification is a popular verification technique that involves capturing design intent in a set of assertions that can be used in formal verification or testing-based checking. However, writing security-centric assertions is a challenging task. In this work, we investigate the use of emerging large language models (LLMs) for code generation in hardware assertion generation for security, where primarily natural language prompts, such as those one would see as code comments in assertion files, are used to produce SystemVerilog assertions. We focus our attention on a popular LLM and characterize its ability to write assertions out of the box, given varying levels of detail in the prompt. We design an evaluation framework that generates a variety of 
    
[^6]: 在联邦学习中平衡隐私与安全：FedGT的群体测试框架

    Balancing Privacy and Security in Federated Learning with FedGT: A Group Testing Framework. (arXiv:2305.05506v1 [cs.LG])

    [http://arxiv.org/abs/2305.05506](http://arxiv.org/abs/2305.05506)

    该论文提出了FedGT框架，通过群体测试的方法在联邦学习中识别并删除恶意客户，从而平衡了隐私和安全，保护数据隐私并提高了识别恶意客户的能力。

    

    我们提出FedGT，一个新颖的框架，用于在联邦学习中识别恶意客户并进行安全聚合。受到群体测试的启发，该框架利用重叠的客户组来检测恶意客户的存在，并通过译码操作识别它们。然后，将这些被识别的客户从模型的训练中删除，并在其余客户之间执行训练。FedGT在隐私和安全之间取得平衡，允许改进识别能力同时仍保护数据隐私。具体而言，服务器学习每个组中客户的聚合模型。通过对MNIST和CIFAR-10数据集进行大量实验，证明了FedGT的有效性，展示了其识别恶意客户的能力，具有低误检和虚警概率，产生高模型效用。

    We propose FedGT, a novel framework for identifying malicious clients in federated learning with secure aggregation. Inspired by group testing, the framework leverages overlapping groups of clients to detect the presence of malicious clients in the groups and to identify them via a decoding operation. The identified clients are then removed from the training of the model, which is performed over the remaining clients. FedGT strikes a balance between privacy and security, allowing for improved identification capabilities while still preserving data privacy. Specifically, the server learns the aggregated model of the clients in each group. The effectiveness of FedGT is demonstrated through extensive experiments on the MNIST and CIFAR-10 datasets, showing its ability to identify malicious clients with low misdetection and false alarm probabilities, resulting in high model utility.
    

