# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SecGPT: An Execution Isolation Architecture for LLM-Based Systems](https://arxiv.org/abs/2403.04960) | 提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题 |
| [^2] | [SoK: Exploring the Potential of Large Language Models for Improving Digital Forensic Investigation Efficiency](https://arxiv.org/abs/2402.19366) | 将大型语言模型（LLMs）整合到数字取证调查中有望提升调查效率，改善可追溯性，并缓解执法机构面临的技术和司法障碍。 |
| [^3] | [Private Fine-tuning of Large Language Models with Zeroth-order Optimization.](http://arxiv.org/abs/2401.04343) | 引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。 |

# 详细

[^1]: SecGPT：一种面向基于LLM系统的执行隔离架构

    SecGPT: An Execution Isolation Architecture for LLM-Based Systems

    [https://arxiv.org/abs/2403.04960](https://arxiv.org/abs/2403.04960)

    提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题

    

    大型语言模型（LLMs）被扩展为系统，如ChatGPT，已经开始支持第三方应用程序。这些LLM应用程序利用LLMs的事实上基于自然语言的自动执行范式：即，应用程序及其交互是用自然语言定义的，提供对用户数据的访问，并被允许自由地相互交互以及与系统互动。这些LLM应用程序生态系统类似于早期计算平台的设置，在那里应用程序和系统之间缺乏足够的隔离。由于第三方应用程序可能不可信，并且受自然语言界面的不精确性加剧，当前的设计会为用户带来安全和隐私风险。在本文中，我们提出了SecGPT，一种面向LLM系统的架构，旨在缓解由第三方应用程序执行引起的安全性和隐私问题。SecGPT的关键思想是隔离应用程序的执行和更多的预

    arXiv:2403.04960v1 Announce Type: cross  Abstract: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more pre
    
[^2]: SoK: 探索大型语言模型在提高数字取证调查效率方面的潜力

    SoK: Exploring the Potential of Large Language Models for Improving Digital Forensic Investigation Efficiency

    [https://arxiv.org/abs/2402.19366](https://arxiv.org/abs/2402.19366)

    将大型语言模型（LLMs）整合到数字取证调查中有望提升调查效率，改善可追溯性，并缓解执法机构面临的技术和司法障碍。

    

    随着需要数字取证分析的案件数量增长，对执法机构及时进行调查的能力产生了担忧。因此，这篇系统化知识论文深入探讨了将大型语言模型（LLMs）整合到数字取证调查中以解决这些挑战的潜力和有效性。对现有的数字取证模型、工具、LLMs、深度学习技术以及在调查中利用LLMs的全面文献综述进行了研究。综述确定了现有数字取证流程中的挑战，并探讨了整合LLMs的障碍和可能性。最终，研究断言，在适当的约束条件下，数字取证中采用LLMs有望提升调查效率，改善可追溯性，并缓解执法机构面临的技术和司法障碍。

    arXiv:2402.19366v1 Announce Type: cross  Abstract: The growing number of cases requiring digital forensic analysis raises concerns about law enforcement's ability to conduct investigations promptly. Consequently, this systemisation of knowledge paper delves into the potential and effectiveness of integrating Large Language Models (LLMs) into digital forensic investigation to address these challenges. A thorough literature review is undertaken, encompassing existing digital forensic models, tools, LLMs, deep learning techniques, and the utilisation of LLMs in investigations. The review identifies current challenges within existing digital forensic processes and explores both the obstacles and possibilities of incorporating LLMs. In conclusion, the study asserts that the adoption of LLMs in digital forensics, with appropriate constraints, holds the potential to enhance investigation efficiency, improve traceability, and alleviate technical and judicial barriers faced by law enforcement e
    
[^3]: 私有零阶优化的大型语言模型的私有微调

    Private Fine-tuning of Large Language Models with Zeroth-order Optimization. (arXiv:2401.04343v1 [cs.LG])

    [http://arxiv.org/abs/2401.04343](http://arxiv.org/abs/2401.04343)

    引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。

    

    在私有数据集上对大型预训练模型进行微调可能会存在违反隐私的风险。差分隐私是一种通过强制算法稳定性来减轻隐私风险的框架。DP-SGD可以以保护隐私的方式训练具有私有数据的模型，但会带来性能损失和重大工程挑战。我们引入了DP-ZO，一种通过私有化零阶优化来保护训练数据隐私的大型语言模型微调方法。我们的方法设计的一个关键见解是，我们使用的零阶算法SPSA中的梯度方向始终是随机的，而仅依赖于私有数据的信息是步长，即一个标量。因此，我们只需要对标量步长进行隐私处理，这是存储效率高的方法。DP-ZO可以使用拉普拉斯噪声或高斯噪声来实现，在不同任务之间提供了隐私和效用之间的强大权衡。

    Fine-tuning large pretrained models on private datasets may run the risk of violating privacy. Differential privacy is a framework for mitigating privacy risks by enforcing algorithmic stability. DP-SGD enables training models with private data in a privacy-preserving manner, but raises new obstacles in the form of performance loss and significant engineering challenges. We introduce DP-ZO, a new method for fine-tuning large language models that preserves the privacy of training data by privatizing zeroth-order optimization. A key insight into the design of our method is that the direction of the gradient in SPSA, the zeroth-order algorithm we use, is always random and the only information that depends on private data is the step size, i.e., a scalar. Therefore, we only need to privatize the scalar step size, which is memory-efficient. DP-ZO, which can be instantiated with either Laplace or Gaussian noise, provides a strong privacy-utility trade-off across different tasks, and model si
    

