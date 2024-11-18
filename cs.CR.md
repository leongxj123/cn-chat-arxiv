# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Security and Privacy Challenges of Large Language Models: A Survey](https://rss.arxiv.org/abs/2402.00888) | 大型语言模型具有卓越的能力，但也面临着安全和隐私攻击的威胁。本调查全面审查了LLM的安全和隐私挑战，涵盖了训练数据、用户和应用风险等方面，并对解决方法进行了回顾。 |
| [^2] | [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710) | 介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。 |
| [^3] | [Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models.](http://arxiv.org/abs/2308.16703) | 本文介绍了故障注入和安全错误攻击用于提取嵌入式神经网络模型的方法，并阐述了对32位微控制器上的深度神经网络进行模型提取攻击的实验结果。 |

# 详细

[^1]: 大型语言模型的安全和隐私挑战：一项调查

    Security and Privacy Challenges of Large Language Models: A Survey

    [https://rss.arxiv.org/abs/2402.00888](https://rss.arxiv.org/abs/2402.00888)

    大型语言模型具有卓越的能力，但也面临着安全和隐私攻击的威胁。本调查全面审查了LLM的安全和隐私挑战，涵盖了训练数据、用户和应用风险等方面，并对解决方法进行了回顾。

    

    大型语言模型（LLM）展示了非凡的能力，并在生成和总结文本、语言翻译和问答等多个领域做出了贡献。如今，LLM正在成为计算机语言处理任务中非常流行的工具，具备分析复杂语言模式并根据上下文提供相关和适当回答的能力。然而，尽管具有显著优势，这些模型也容易受到安全和隐私攻击的威胁，如越狱攻击、数据污染攻击和个人可识别信息泄露攻击。本调查全面审查了LLM的安全和隐私挑战，包括训练数据和用户方面的问题，以及在交通、教育和医疗等各个领域中应用带来的风险。我们评估了LLM的脆弱性程度，调查了出现的安全和隐私攻击，并对潜在的解决方法进行了回顾。

    Large Language Models (LLMs) have demonstrated extraordinary capabilities and contributed to multiple fields, such as generating and summarizing text, language translation, and question-answering. Nowadays, LLM is becoming a very popular tool in computerized language processing tasks, with the capability to analyze complicated linguistic patterns and provide relevant and appropriate responses depending on the context. While offering significant advantages, these models are also vulnerable to security and privacy attacks, such as jailbreaking attacks, data poisoning attacks, and Personally Identifiable Information (PII) leakage attacks. This survey provides a thorough review of the security and privacy challenges of LLMs for both training data and users, along with the application-based risks in various domains, such as transportation, education, and healthcare. We assess the extent of LLM vulnerabilities, investigate emerging security and privacy attacks for LLMs, and review the potent
    
[^2]: 基于优化的对LLM评判系统的提示注入攻击

    Optimization-based Prompt Injection Attack to LLM-as-a-Judge

    [https://arxiv.org/abs/2403.17710](https://arxiv.org/abs/2403.17710)

    介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。

    

    LLM-as-a-Judge 是一种可以使用大型语言模型（LLMs）评估文本信息的新颖解决方案。根据现有研究，LLMs在提供传统人类评估的引人注目替代方面表现出色。然而，这些系统针对提示注入攻击的鲁棒性仍然是一个未解决的问题。在这项工作中，我们引入了JudgeDeceiver，一种针对LLM-as-a-Judge量身定制的基于优化的提示注入攻击。我们的方法制定了一个精确的优化目标，用于攻击LLM-as-a-Judge的决策过程，并利用优化算法高效地自动化生成对抗序列，实现对模型评估的有针对性和有效的操作。与手工制作的提示注入攻击相比，我们的方法表现出卓越的功效，给基于LLM的判断系统当前的安全范式带来了重大挑战。

    arXiv:2403.17710v1 Announce Type: cross  Abstract: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. T
    
[^3]: 故障注入和安全错误攻击用于提取嵌入式神经网络模型

    Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models. (arXiv:2308.16703v1 [cs.CR])

    [http://arxiv.org/abs/2308.16703](http://arxiv.org/abs/2308.16703)

    本文介绍了故障注入和安全错误攻击用于提取嵌入式神经网络模型的方法，并阐述了对32位微控制器上的深度神经网络进行模型提取攻击的实验结果。

    

    模型提取作为一种关键的安全威胁而出现，攻击向量利用了算法和实现方面的方法。攻击者的主要目标是尽可能多地窃取受保护的受害者模型的信息，以便他可以用替代模型来模仿它，即使只有有限的访问相似的训练数据。最近，物理攻击，如故障注入，已经显示出对嵌入式模型的完整性和机密性的令人担忧的效果。我们的重点是32位微控制器上的嵌入式深度神经网络模型，这是物联网中广泛使用的硬件平台系列，以及使用标准故障注入策略-安全错误攻击（SEA）来进行具有有限训练数据访问的模型提取攻击。由于攻击强烈依赖于输入查询，我们提出了一种黑盒方法来构建一个成功的攻击集。对于一个经典的卷积神经网络，我们成功地恢复了至少90%的

    Model extraction emerges as a critical security threat with attack vectors exploiting both algorithmic and implementation-based approaches. The main goal of an attacker is to steal as much information as possible about a protected victim model, so that he can mimic it with a substitute model, even with a limited access to similar training data. Recently, physical attacks such as fault injection have shown worrying efficiency against the integrity and confidentiality of embedded models. We focus on embedded deep neural network models on 32-bit microcontrollers, a widespread family of hardware platforms in IoT, and the use of a standard fault injection strategy - Safe Error Attack (SEA) - to perform a model extraction attack with an adversary having a limited access to training data. Since the attack strongly depends on the input queries, we propose a black-box approach to craft a successful attack set. For a classical convolutional neural network, we successfully recover at least 90% of
    

