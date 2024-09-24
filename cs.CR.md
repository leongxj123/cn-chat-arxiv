# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond the Request: Harnessing HTTP Response Headers for Cross-Browser Web Tracker Classification in an Imbalanced Setting](https://rss.arxiv.org/abs/2402.01240) | 本研究通过利用HTTP响应头设计了机器学习分类器，在跨浏览器环境下有效检测Web追踪器，结果在Chrome和Firefox上表现出较高的准确性和性能。 |
| [^2] | [Institutional Platform for Secure Self-Service Large Language Model Exploration](https://rss.arxiv.org/abs/2402.00913) | 这个论文介绍了一个用户友好型平台，旨在使大型定制语言模型更易于使用，通过最新的多LoRA推理技术和定制适配器，实现了数据隔离、加密和身份验证的安全服务。 |
| [^3] | [Membership Inference Attacks and Privacy in Topic Modeling](https://arxiv.org/abs/2403.04451) | 主题建模中提出了会员推理攻击，通过差分隐私词汇选择来改善隐私风险 |
| [^4] | [Stop Reasoning! When Multimodal LLMs with Chain-of-Thought Reasoning Meets Adversarial Images](https://arxiv.org/abs/2402.14899) | 该研究评估了多模态LLMs在采用串联推理时的对抗鲁棒性，发现串联推理在一定程度上提高了对抗性鲁棒性，但引入了一种新的停止推理攻击技术成功规避了这种增强。 |
| [^5] | [Automated Security Response through Online Learning with Adaptive Conjectures](https://arxiv.org/abs/2402.12499) | 该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。 |
| [^6] | [RAMP: Boosting Adversarial Robustness Against Multiple $l_p$ Perturbations](https://arxiv.org/abs/2402.06827) | 该论文提出了一种名为RAMP的框架，旨在增强对多个$l_p$扰动的对抗鲁棒性。通过分析不同$l_p$攻击之间的权衡关系，并设计逻辑配对损失来提高准确性和鲁棒性的平衡。同时，通过将自然训练与对抗训练相结合，整合有用信息以调和准确性和鲁棒性的权衡。 |
| [^7] | [LLM in the Shell: Generative Honeypots.](http://arxiv.org/abs/2309.00155) | 本研究引入了一种基于大型语言模型的新方法来创建动态和真实的软件蜜罐，解决了以往蜜罐的重要局限性，并通过实验验证了其高准确率。 |

# 详细

[^1]: 超越请求：利用HTTP响应头在不平衡环境中进行跨浏览器Web追踪器分类

    Beyond the Request: Harnessing HTTP Response Headers for Cross-Browser Web Tracker Classification in an Imbalanced Setting

    [https://rss.arxiv.org/abs/2402.01240](https://rss.arxiv.org/abs/2402.01240)

    本研究通过利用HTTP响应头设计了机器学习分类器，在跨浏览器环境下有效检测Web追踪器，结果在Chrome和Firefox上表现出较高的准确性和性能。

    

    万维网的连通性主要归因于HTTP协议，其中的HTTP消息提供了有关网络安全和隐私的信息头字段，特别是关于Web追踪。尽管已有研究利用HTTP/S请求消息来识别Web追踪器，但往往忽视了HTTP/S响应头。本研究旨在设计使用HTTP/S响应头进行Web追踪器检测的有效机器学习分类器。通过浏览器扩展程序T.EX获取的Chrome、Firefox和Brave浏览器的数据作为我们的数据集。在Chrome数据上训练了11个监督模型，并在所有浏览器上进行了测试。结果表明，在Chrome和Firefox上具有高准确性、F1分数、精确度、召回率和最小对数损失误差的性能，但在Brave浏览器上表现不佳，可能是由于其不同的数据分布和特征集。研究表明，这些分类器可以用于检测Web追踪器。

    The World Wide Web's connectivity is greatly attributed to the HTTP protocol, with HTTP messages offering informative header fields that appeal to disciplines like web security and privacy, especially concerning web tracking. Despite existing research employing HTTP/S request messages to identify web trackers, HTTP/S response headers are often overlooked. This study endeavors to design effective machine learning classifiers for web tracker detection using HTTP/S response headers. Data from the Chrome, Firefox, and Brave browsers, obtained through the traffic monitoring browser extension T.EX, serves as our data set. Eleven supervised models were trained on Chrome data and tested across all browsers. The results demonstrated high accuracy, F1-score, precision, recall, and minimal log-loss error for Chrome and Firefox, but subpar performance on Brave, potentially due to its distinct data distribution and feature set. The research suggests that these classifiers are viable for detecting w
    
[^2]: 用于安全自助大型语言模型探索的机构平台

    Institutional Platform for Secure Self-Service Large Language Model Exploration

    [https://rss.arxiv.org/abs/2402.00913](https://rss.arxiv.org/abs/2402.00913)

    这个论文介绍了一个用户友好型平台，旨在使大型定制语言模型更易于使用，通过最新的多LoRA推理技术和定制适配器，实现了数据隔离、加密和身份验证的安全服务。

    

    本文介绍了由肯塔基大学应用人工智能中心开发的用户友好型平台，旨在使大型定制语言模型（LLM）更易于使用。通过利用最近在多LoRA推理方面的进展，系统有效地适应了各类用户和项目的定制适配器。论文概述了系统的架构和关键特性，包括数据集策划、模型训练、安全推理和基于文本的特征提取。我们通过使用基于代理的方法建立了一个基于租户意识的计算网络，在安全地利用孤立资源岛的基础上形成了一个统一的系统。该平台致力于提供安全的LLM服务，强调过程和数据隔离、端到端加密以及基于角色的资源身份验证。该贡献与实现简化访问先进的AI模型和技术以支持科学发现的总体目标一致。

    This paper introduces a user-friendly platform developed by the University of Kentucky Center for Applied AI, designed to make large, customized language models (LLMs) more accessible. By capitalizing on recent advancements in multi-LoRA inference, the system efficiently accommodates custom adapters for a diverse range of users and projects. The paper outlines the system's architecture and key features, encompassing dataset curation, model training, secure inference, and text-based feature extraction.   We illustrate the establishment of a tenant-aware computational network using agent-based methods, securely utilizing islands of isolated resources as a unified system. The platform strives to deliver secure LLM services, emphasizing process and data isolation, end-to-end encryption, and role-based resource authentication. This contribution aligns with the overarching goal of enabling simplified access to cutting-edge AI models and technology in support of scientific discovery.
    
[^3]: 会员推理攻击与主题建模中的隐私

    Membership Inference Attacks and Privacy in Topic Modeling

    [https://arxiv.org/abs/2403.04451](https://arxiv.org/abs/2403.04451)

    主题建模中提出了会员推理攻击，通过差分隐私词汇选择来改善隐私风险

    

    最近的研究表明，大型语言模型容易受到推理训练数据方面的隐私攻击。然而，目前还不清楚更简单的生成模型，例如主题模型，是否存在类似的漏洞。在这项工作中，我们提出了一种针对主题模型的攻击，可以自信地识别Latent Dirichlet Allocation中训练数据的成员。我们的结果表明，与大型神经模型相关联的隐私风险并不仅限于大型神经模型。此外，为了减轻这些漏洞，我们探讨了差分隐私（DP）主题建模。我们提出了一个私密主题建模框架，将DP词汇选择作为预处理步骤，并展示它不仅改善了隐私性，而且在实用性方面的影响有限。

    arXiv:2403.04451v1 Announce Type: cross  Abstract: Recent research shows that large language models are susceptible to privacy attacks that infer aspects of the training data. However, it is unclear if simpler generative models, like topic models, share similar vulnerabilities. In this work, we propose an attack against topic models that can confidently identify members of the training data in Latent Dirichlet Allocation. Our results suggest that the privacy risks associated with generative modeling are not restricted to large neural models. Additionally, to mitigate these vulnerabilities, we explore differentially private (DP) topic modeling. We propose a framework for private topic modeling that incorporates DP vocabulary selection as a pre-processing step, and show that it improves privacy while having limited effects on practical utility.
    
[^4]: 停止推理！当多模态LLMs与串联推理遇到对抗性图像

    Stop Reasoning! When Multimodal LLMs with Chain-of-Thought Reasoning Meets Adversarial Images

    [https://arxiv.org/abs/2402.14899](https://arxiv.org/abs/2402.14899)

    该研究评估了多模态LLMs在采用串联推理时的对抗鲁棒性，发现串联推理在一定程度上提高了对抗性鲁棒性，但引入了一种新的停止推理攻击技术成功规避了这种增强。

    

    最近，多模态LLMs（MLLMs）展示了很强的理解图像的能力。然而，像传统视觉模型一样，它们仍然容易受到对抗性图像的攻击。与此同时，串联推理（CoT）已经被广泛应用在MLLMs上，不仅提高了模型的性能，而且通过提供中间推理步骤来增强模型的可解释性。然而，目前还缺乏关于MLLMs在CoT下的对抗鲁棒性的研究，以及在MLLMs用对抗性图像推断错误答案时推理的合理性。我们的研究评估了采用CoT推理时MLLMs的对抗鲁棒性，发现CoT在一定程度上提高了对抗性鲁棒性，抵抗了已有的攻击方法。此外，我们引入了一种新的停止推理攻击技术，可以有效地规避CoT引起的鲁棒性增强。最后，我们展示了CoT推理的变化。

    arXiv:2402.14899v1 Announce Type: cross  Abstract: Recently, Multimodal LLMs (MLLMs) have shown a great ability to understand images. However, like traditional vision models, they are still vulnerable to adversarial images. Meanwhile, Chain-of-Thought (CoT) reasoning has been widely explored on MLLMs, which not only improves model's performance, but also enhances model's explainability by giving intermediate reasoning steps. Nevertheless, there is still a lack of study regarding MLLMs' adversarial robustness with CoT and an understanding of what the rationale looks like when MLLMs infer wrong answers with adversarial images. Our research evaluates the adversarial robustness of MLLMs when employing CoT reasoning, finding that CoT marginally improves adversarial robustness against existing attack methods. Moreover, we introduce a novel stop-reasoning attack technique that effectively bypasses the CoT-induced robustness enhancements. Finally, we demonstrate the alterations in CoT reasonin
    
[^5]: 通过自适应猜想的在线学习实现自动化安全响应

    Automated Security Response through Online Learning with Adaptive Conjectures

    [https://arxiv.org/abs/2402.12499](https://arxiv.org/abs/2402.12499)

    该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。

    

    我们研究了针对IT基础设施的自动化安全响应，并将攻击者和防御者之间的互动形式表述为一个部分观测、非平稳博弈。我们放宽了游戏模型正确规定的标准假设，并考虑每个参与者对模型有一个概率性猜想，可能在某种意义上错误规定，即真实模型的概率为0。这种形式允许我们捕捉关于基础设施和参与者意图的不确定性。为了在线学习有效的游戏策略，我们设计了一种新颖的方法，其中一个参与者通过贝叶斯学习迭代地调整其猜想，并通过推演更新其策略。我们证明了猜想会收敛到最佳拟合，并提供了在具有猜测模型的情况下推演实现性能改进的上限。为了刻画游戏的稳定状态，我们提出了Berk-Nash平衡的一个变种。

    arXiv:2402.12499v1 Announce Type: cross  Abstract: We study automated security response for an IT infrastructure and formulate the interaction between an attacker and a defender as a partially observed, non-stationary game. We relax the standard assumption that the game model is correctly specified and consider that each player has a probabilistic conjecture about the model, which may be misspecified in the sense that the true model has probability 0. This formulation allows us to capture uncertainty about the infrastructure and the intents of the players. To learn effective game strategies online, we design a novel method where a player iteratively adapts its conjecture using Bayesian learning and updates its strategy through rollout. We prove that the conjectures converge to best fits, and we provide a bound on the performance improvement that rollout enables with a conjectured model. To characterize the steady state of the game, we propose a variant of the Berk-Nash equilibrium. We 
    
[^6]: RAMP：增强对多个$l_p$扰动的对抗鲁棒性

    RAMP: Boosting Adversarial Robustness Against Multiple $l_p$ Perturbations

    [https://arxiv.org/abs/2402.06827](https://arxiv.org/abs/2402.06827)

    该论文提出了一种名为RAMP的框架，旨在增强对多个$l_p$扰动的对抗鲁棒性。通过分析不同$l_p$攻击之间的权衡关系，并设计逻辑配对损失来提高准确性和鲁棒性的平衡。同时，通过将自然训练与对抗训练相结合，整合有用信息以调和准确性和鲁棒性的权衡。

    

    在提高对单个$l_p$范数受限的对抗攻击的鲁棒性方面，已经有相当多的工作在使用对抗训练（AT）进行研究。然而，AT模型的多范数鲁棒性（共同准确性）仍然较低。我们观察到，同时获得良好的共同准确性和清洁准确性是困难的，因为在多个$l_p$扰动之间存在鲁棒性、准确性/鲁棒性/效率之间的权衡。通过从分布转变的角度分析这些权衡，我们确定了$l_p$攻击之间的关键权衡对，以提高效率并设计了一个逻辑配对损失来提高共同准确性。接下来，我们通过梯度投影将自然训练与AT相连接，以从自然训练中找到并整合有用的信息到AT中，从而调和准确性/鲁棒性的权衡。结合我们的贡献，我们提出了一个名为\textbf{RAMP}的框架，来提高对多个$l_p$扰动的鲁棒性。我们展示了\textbf{RAMP}可以很容易地适应...

    There is considerable work on improving robustness against adversarial attacks bounded by a single $l_p$ norm using adversarial training (AT). However, the multiple-norm robustness (union accuracy) of AT models is still low. We observe that simultaneously obtaining good union and clean accuracy is hard since there are tradeoffs between robustness against multiple $l_p$ perturbations, and accuracy/robustness/efficiency. By analyzing the tradeoffs from the lens of distribution shifts, we identify the key tradeoff pair among $l_p$ attacks to boost efficiency and design a logit pairing loss to improve the union accuracy. Next, we connect natural training with AT via gradient projection, to find and incorporate useful information from natural training into AT, which moderates the accuracy/robustness tradeoff. Combining our contributions, we propose a framework called \textbf{RAMP}, to boost the robustness against multiple $l_p$ perturbations. We show \textbf{RAMP} can be easily adapted for 
    
[^7]: LLM在Shell中的应用：生成式蜜罐

    LLM in the Shell: Generative Honeypots. (arXiv:2309.00155v1 [cs.CR])

    [http://arxiv.org/abs/2309.00155](http://arxiv.org/abs/2309.00155)

    本研究引入了一种基于大型语言模型的新方法来创建动态和真实的软件蜜罐，解决了以往蜜罐的重要局限性，并通过实验验证了其高准确率。

    

    蜜罐是网络安全中的重要工具。然而，大多数蜜罐（即使是高交互式的）缺乏足够的真实感来欺骗攻击者。这个限制使得它们很容易被识别，从而影响到它们的有效性。本研究引入了一种基于大型语言模型的新方法来创建动态和真实的软件蜜罐。初步结果表明，LLM能够创建可信且动态的蜜罐，能够解决以往蜜罐的重要局限性，如确定性响应、缺乏适应性等。我们通过与需要判断蜜罐回应是否虚假的攻击者进行实验来评估每个命令的真实性。我们提出的蜜罐，称为shelLM，达到了0.92的准确率。

    Honeypots are essential tools in cybersecurity. However, most of them (even the high-interaction ones) lack the required realism to engage and fool human attackers. This limitation makes them easily discernible, hindering their effectiveness. This work introduces a novel method to create dynamic and realistic software honeypots based on Large Language Models. Preliminary results indicate that LLMs can create credible and dynamic honeypots capable of addressing important limitations of previous honeypots, such as deterministic responses, lack of adaptability, etc. We evaluated the realism of each command by conducting an experiment with human attackers who needed to say if the answer from the honeypot was fake or not. Our proposed honeypot, called shelLM, reached an accuracy rate of 0.92.
    

