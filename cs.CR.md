# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging AI Planning For Detecting Cloud Security Vulnerabilities](https://arxiv.org/abs/2402.10985) | 提出了一个通用框架来建模云系统中的访问控制策略，并开发了基于PDDL模型的新方法来检测可能导致诸如勒索软件和敏感数据外泄等广泛攻击的安全漏洞。 |
| [^2] | [SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding](https://arxiv.org/abs/2402.08983) | 本文提出了一种名为SafeDecoding的解决方案，通过安全感知解码策略来防御大型语言模型（LLMs）的越狱攻击。该策略可以生成对用户查询有益且无害的响应，有效缓解了LLMs安全性威胁。 |
| [^3] | [Coordinated Disclosure for AI: Beyond Security Vulnerabilities](https://arxiv.org/abs/2402.07039) | 这篇论文提出了一种针对机器学习和人工智能问题的协调缺陷披露（CFD）框架，以解决目前领域中缺乏结构化过程的问题。 |
| [^4] | [The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks](https://arxiv.org/abs/2402.06357) | 本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。 |
| [^5] | [The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks.](http://arxiv.org/abs/2310.15469) | 《Janus接口：大型语言模型微调如何放大隐私风险》研究了大型语言模型的微调对个人信息泄露的风险，发现了一种新的LLM利用途径。 |

# 详细

[^1]: 利用AI规划技术检测云安全漏洞

    Leveraging AI Planning For Detecting Cloud Security Vulnerabilities

    [https://arxiv.org/abs/2402.10985](https://arxiv.org/abs/2402.10985)

    提出了一个通用框架来建模云系统中的访问控制策略，并开发了基于PDDL模型的新方法来检测可能导致诸如勒索软件和敏感数据外泄等广泛攻击的安全漏洞。

    

    云计算服务提供了可扩展且具有成本效益的数据存储、处理和协作解决方案。随着它们的普及，与其安全漏洞相关的担忧也在增长，这可能导致数据泄露和勒索软件等复杂攻击。为了应对这些问题，我们首先提出了一个通用框架，用于表达云系统中不同对象（如用户、数据存储、安全角色）之间的关系，以建模云系统中的访问控制策略。访问控制误配置通常是云攻击的主要原因。其次，我们开发了一个PDDL模型，用于检测安全漏洞，例如可能导致广泛攻击（如勒索软件）和敏感数据外泄等。规划器可以生成攻击以识别云中的此类漏洞。最后，我们在14个不同商业组织的真实亚马逊AWS云配置上测试了我们的方法。

    arXiv:2402.10985v1 Announce Type: cross  Abstract: Cloud computing services provide scalable and cost-effective solutions for data storage, processing, and collaboration. Alongside their growing popularity, concerns related to their security vulnerabilities leading to data breaches and sophisticated attacks such as ransomware are growing. To address these, first, we propose a generic framework to express relations between different cloud objects such as users, datastores, security roles, to model access control policies in cloud systems. Access control misconfigurations are often the primary driver for cloud attacks. Second, we develop a PDDL model for detecting security vulnerabilities which can for example lead to widespread attacks such as ransomware, sensitive data exfiltration among others. A planner can then generate attacks to identify such vulnerabilities in the cloud. Finally, we test our approach on 14 real Amazon AWS cloud configurations of different commercial organizations
    
[^2]: SafeDecoding: 通过安全感知解码防御越狱攻击

    SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding

    [https://arxiv.org/abs/2402.08983](https://arxiv.org/abs/2402.08983)

    本文提出了一种名为SafeDecoding的解决方案，通过安全感知解码策略来防御大型语言模型（LLMs）的越狱攻击。该策略可以生成对用户查询有益且无害的响应，有效缓解了LLMs安全性威胁。

    

    随着大型语言模型（LLMs）越来越多地应用于代码生成和聊天机器人辅助等现实应用中，人们为了使LLM的行为与人类价值观保持一致，包括安全性在内做出了大量努力。越狱攻击旨在引发LLM的非预期和不安全行为，仍然是LLM安全性的重要威胁。本文旨在通过引入SafeDecoding来防御LLM的越狱攻击，这是一种安全感知的解码策略，用于生成对用户查询有益且无害的响应。我们在开发SafeDecoding时的洞察力基于观察到，即使代表有害内容的标记的概率超过代表无害响应的标记的概率，安全免责声明仍然出现在按概率降序排序的标记中的前几个。这使我们能够通过识别安全免责声明并增强其良性影响力来减轻越狱攻击。

    arXiv:2402.08983v1 Announce Type: cross Abstract: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their
    
[^3]: 人工智能的协调披露：超越安全漏洞

    Coordinated Disclosure for AI: Beyond Security Vulnerabilities

    [https://arxiv.org/abs/2402.07039](https://arxiv.org/abs/2402.07039)

    这篇论文提出了一种针对机器学习和人工智能问题的协调缺陷披露（CFD）框架，以解决目前领域中缺乏结构化过程的问题。

    

    目前，人工智能领域的伤害报告在披露或解决算法缺陷方面仍然是一种临时性的操作，缺乏结构化的过程。相比之下，协调漏洞披露（CVD）的伦理和生态系统在软件安全和透明度方面发挥着关键作用。在美国的背景下，为了鼓励秉持善意行事的安全研究人员，建立一个安全防护条款以对抗计算机欺诈和滥用法案一直存在长期的法律和政策斗争。值得注意的是，机器学习（ML）模型中的算法缺陷与传统软件漏洞存在着不同的挑战，需要一种专门的方法。为了解决这一差距，我们提出了一种针对机器学习和人工智能问题特殊复杂性的专门协调缺陷披露（CFD）框架的实施。本文深入研究了ML中的披露历史背景，包括

    Harm reporting in the field of Artificial Intelligence (AI) currently operates on an ad hoc basis, lacking a structured process for disclosing or addressing algorithmic flaws. In contrast, the Coordinated Vulnerability Disclosure (CVD) ethos and ecosystem play a pivotal role in software security and transparency. Within the U.S. context, there has been a protracted legal and policy struggle to establish a safe harbor from the Computer Fraud and Abuse Act, aiming to foster institutional support for security researchers acting in good faith. Notably, algorithmic flaws in Machine Learning (ML) models present distinct challenges compared to traditional software vulnerabilities, warranting a specialized approach. To address this gap, we propose the implementation of a dedicated Coordinated Flaw Disclosure (CFD) framework tailored to the intricacies of machine learning and artificial intelligence issues. This paper delves into the historical landscape of disclosures in ML, encompassing the a
    
[^4]: SpongeNet 攻击：深度神经网络的海绵权重中毒

    The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks

    [https://arxiv.org/abs/2402.06357](https://arxiv.org/abs/2402.06357)

    本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。

    

    海绵攻击旨在增加在硬件加速器上部署的神经网络的能耗和计算时间。现有的海绵攻击可以通过海绵示例进行推理，也可以通过海绵中毒在训练过程中进行。海绵示例利用添加到模型输入的扰动来增加能量和延迟，而海绵中毒则改变模型的目标函数来引发推理时的能量/延迟效应。在这项工作中，我们提出了一种新颖的海绵攻击，称为 SpongeNet。SpongeNet 是第一个直接作用于预训练模型参数的海绵攻击。我们的实验表明，相比于海绵中毒，SpongeNet 可以成功增加视觉模型的能耗，并且所需的样本数量更少。我们的实验结果表明，如果不专门针对海绵中毒进行调整（即减小批归一化偏差值），则毒害防御会失效。我们的工作显示出海绵攻击的影响。

    Sponge attacks aim to increase the energy consumption and computation time of neural networks deployed on hardware accelerators. Existing sponge attacks can be performed during inference via sponge examples or during training via Sponge Poisoning. Sponge examples leverage perturbations added to the model's input to increase energy and latency, while Sponge Poisoning alters the objective function of a model to induce inference-time energy/latency effects.   In this work, we propose a novel sponge attack called SpongeNet. SpongeNet is the first sponge attack that is performed directly on the parameters of a pre-trained model. Our experiments show that SpongeNet can successfully increase the energy consumption of vision models with fewer samples required than Sponge Poisoning. Our experiments indicate that poisoning defenses are ineffective if not adjusted specifically for the defense against Sponge Poisoning (i.e., they decrease batch normalization bias values). Our work shows that Spong
    
[^5]: 《Janus接口：大型语言模型微调如何放大隐私风险》

    The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks. (arXiv:2310.15469v1 [cs.CR])

    [http://arxiv.org/abs/2310.15469](http://arxiv.org/abs/2310.15469)

    《Janus接口：大型语言模型微调如何放大隐私风险》研究了大型语言模型的微调对个人信息泄露的风险，发现了一种新的LLM利用途径。

    

    2018年后的时代标志着大型语言模型（LLM）的出现，OpenAI的ChatGPT等创新展示了惊人的语言能力。随着行业在增加模型参数并利用大量的人类语言数据方面的努力，安全和隐私挑战也出现了。其中最重要的是在基于网络的数据获取过程中，可能会意外积累个人可识别信息（PII），从而导致意外的PII泄露风险。虽然像RLHF和灾难性遗忘这样的策略已被用来控制隐私侵权的风险，但LLM的最新进展（以OpenAI的GPT-3.5的微调界面为代表）重新引发了关注。有人可能会问：LLM的微调是否会导致训练数据集中嵌入的个人信息泄漏？本文报道了首次尝试寻求答案的努力，重点是我们发现了一种新的LLM利用途径。

    The era post-2018 marked the advent of Large Language Models (LLMs), with innovations such as OpenAI's ChatGPT showcasing prodigious linguistic prowess. As the industry galloped toward augmenting model parameters and capitalizing on vast swaths of human language data, security and privacy challenges also emerged. Foremost among these is the potential inadvertent accrual of Personal Identifiable Information (PII) during web-based data acquisition, posing risks of unintended PII disclosure. While strategies like RLHF during training and Catastrophic Forgetting have been marshaled to control the risk of privacy infringements, recent advancements in LLMs, epitomized by OpenAI's fine-tuning interface for GPT-3.5, have reignited concerns. One may ask: can the fine-tuning of LLMs precipitate the leakage of personal information embedded within training datasets? This paper reports the first endeavor to seek the answer to the question, particularly our discovery of a new LLM exploitation avenue
    

