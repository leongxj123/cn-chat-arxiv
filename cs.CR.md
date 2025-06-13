# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices](https://arxiv.org/abs/2403.12503) | 该论文全面调查了与大型语言模型相关的安全和隐私问题，并提出了未来研究的有前途方向。 |
| [^2] | [PRSA: Prompt Reverse Stealing Attacks against Large Language Models](https://arxiv.org/abs/2402.19200) | 本文提出了针对商业LLMs的提示反窃取攻击框架PRSA，通过分析输入-输出对的关键特征实现攻击。 |
| [^3] | [IoTGeM: Generalizable Models for Behaviour-Based IoT Attack Detection.](http://arxiv.org/abs/2401.01343) | 本研究提出了一种通用模型，用于基于行为的物联网攻击检测。该模型通过改进滚动窗口特征提取、引入多步骤特征选择、使用隔离的训练和测试数据集以及使用可解释的人工智能技术来提高检测和性能，并且经过严格评估。 |

# 详细

[^1]: 保护大型语言模型：威胁、漏洞和负责任的实践

    Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices

    [https://arxiv.org/abs/2403.12503](https://arxiv.org/abs/2403.12503)

    该论文全面调查了与大型语言模型相关的安全和隐私问题，并提出了未来研究的有前途方向。

    

    大型语言模型(LLMs)显著改变了自然语言处理(NLP)的格局。它们对各种任务产生了影响，从而彻底改变了我们处理语言理解和生成的方式。然而，除了它们引人注目的实用性外，LLMs还带来了重要的安全和风险考虑。这些挑战需要仔细研究，以确保负责任的部署，并防范潜在的漏洞。本研究全面调查了与LLMs相关的安全和隐私问题，从五个主题角度进行：安全和隐私问题、对抗性攻击的漏洞、LLMs误用可能造成的潜在危害、缓解策略以解决这些挑战，同时识别当前策略的局限性。最后，本文建议未来研究的有前途的方向，以增强LLMs的安全和风险管理。

    arXiv:2403.12503v1 Announce Type: cross  Abstract: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.
    
[^2]: PRSA：大型语言模型的提示反盗窃攻击

    PRSA: Prompt Reverse Stealing Attacks against Large Language Models

    [https://arxiv.org/abs/2402.19200](https://arxiv.org/abs/2402.19200)

    本文提出了针对商业LLMs的提示反窃取攻击框架PRSA，通过分析输入-输出对的关键特征实现攻击。

    

    提示作为重要的知识产权，使得大型语言模型（LLMs）能够执行特定任务而无需微调，突显了它们不断增长的重要性。随着基于提示的服务的崛起，如提示市场和LLM应用程序，提供者经常通过输入-输出示例展示提示的能力，以吸引用户。然而，这种范式提出了一个关键的安全问题：暴露输入-输出对是否会对潜在提示泄漏构成风险，侵犯开发者的知识产权？就我们所知，这个问题还没有得到全面探讨。为了弥补这一空白，在本文中，我们进行了首次深入探讨，并提出了一个针对商业LLMs的提示反窃取攻击框架，即PRSA。PRSA的主要思想是通过分析输入-输出对的关键特征，我们模仿并g

    arXiv:2402.19200v1 Announce Type: cross  Abstract: Prompt, recognized as crucial intellectual property, enables large language models (LLMs) to perform specific tasks without the need of fine-tuning, underscoring their escalating importance. With the rise of prompt-based services, such as prompt marketplaces and LLM applications, providers often display prompts' capabilities through input-output examples to attract users. However, this paradigm raises a pivotal security concern: does the exposure of input-output pairs pose the risk of potential prompt leakage, infringing on the intellectual property rights of the developers? To our knowledge, this problem still has not been comprehensively explored yet. To remedy this gap, in this paper, we perform the first in depth exploration and propose a novel attack framework for reverse-stealing prompts against commercial LLMs, namely PRSA. The main idea of PRSA is that by analyzing the critical features of the input-output pairs, we mimic and g
    
[^3]: IoTGeM: 用于基于行为的物联网攻击检测的通用模型

    IoTGeM: Generalizable Models for Behaviour-Based IoT Attack Detection. (arXiv:2401.01343v1 [cs.CR])

    [http://arxiv.org/abs/2401.01343](http://arxiv.org/abs/2401.01343)

    本研究提出了一种通用模型，用于基于行为的物联网攻击检测。该模型通过改进滚动窗口特征提取、引入多步骤特征选择、使用隔离的训练和测试数据集以及使用可解释的人工智能技术来提高检测和性能，并且经过严格评估。

    

    过去关于物联网设备网络的基于行为的攻击检测研究，所得到的机器学习模型的适应能力有限，并且往往没有得到证明。本文提出了一种针对建模物联网网络攻击的方法，着重于通用性，同时也能提高检测和性能。首先，我们提出了一种改进的滚动窗口特征提取方法，并引入了一个多步骤特征选择过程来减少过拟合。其次，我们使用隔离的训练和测试数据集来构建和测试模型，从而避免了先前模型在通用性方面的常见数据泄漏问题。第三，我们使用多样化的机器学习模型、评估指标和数据集对我们的方法进行了严格评估。最后，我们使用可解释的人工智能技术来增加模型的可信度，从而能够识别出支撑攻击准确检测的特征。

    Previous research on behaviour-based attack detection on networks of IoT devices has resulted in machine learning models whose ability to adapt to unseen data is limited, and often not demonstrated. In this paper we present an approach for modelling IoT network attacks that focuses on generalizability, yet also leads to better detection and performance. First, we present an improved rolling window approach for feature extraction, and introduce a multi-step feature selection process that reduces overfitting. Second, we build and test models using isolated train and test datasets, thereby avoiding common data leaks that have limited the generalizability of previous models. Third, we rigorously evaluate our methodology using a diverse portfolio of machine learning models, evaluation metrics and datasets. Finally, we build confidence in the models by using explainable AI techniques, allowing us to identify the features that underlie accurate detection of attacks.
    

