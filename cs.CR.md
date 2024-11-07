# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Whispers in the Machine: Confidentiality in LLM-integrated Systems](https://arxiv.org/abs/2402.06922) | 本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。 |
| [^2] | [Teach Large Language Models to Forget Privacy.](http://arxiv.org/abs/2401.00870) | 这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。 |

# 详细

[^1]: 机器中的私语：LLM集成系统中的保密性

    Whispers in the Machine: Confidentiality in LLM-integrated Systems

    [https://arxiv.org/abs/2402.06922](https://arxiv.org/abs/2402.06922)

    本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。

    

    大规模语言模型（LLM）越来越多地与外部工具集成。尽管这些集成可以显著提高LLM的功能，但它们也在不同组件之间创建了一个新的攻击面，可能泄露机密数据。具体而言，恶意工具可以利用LLM本身的漏洞来操纵模型并损害其他服务的数据，这引发了在LLM集成环境中如何保护私密数据的问题。在这项工作中，我们提供了一种系统评估LLM集成系统保密性的方法。为此，我们形式化了一个"秘密密钥"游戏，可以捕捉模型隐藏私人信息的能力。这使我们能够比较模型对保密性攻击的脆弱性以及不同防御策略的有效性。在这个框架中，我们评估了八种先前发表的攻击和四种防御方法。我们发现当前的防御方法缺乏泛化性能。

    Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.   In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization
    
[^2]: 教导大型语言模型忘记隐私

    Teach Large Language Models to Forget Privacy. (arXiv:2401.00870v1 [cs.CR])

    [http://arxiv.org/abs/2401.00870](http://arxiv.org/abs/2401.00870)

    这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。

    

    大型语言模型（LLM）已被证明具有强大的能力，但隐私泄露的风险仍然是一个重要问题。传统的保护隐私方法，如差分隐私和同态加密，在只有黑盒API的环境下是不足够的，要求模型透明性或大量计算资源。我们提出了Prompt2Forget（P2F），这是第一个设计用于解决LLM本地隐私挑战的框架，通过教导LLM忘记来实现。该方法涉及将完整问题分解为较小的片段，生成虚构的答案，并使模型对原始输入的记忆模糊化。我们根据不同领域的包含隐私敏感信息的问题创建了基准数据集。P2F实现了零-shot泛化，可以在多种应用场景下自适应，无需手动调整。实验结果表明，P2F具有很强的模糊化LLM记忆的能力，而不会损失任何实用性。

    Large Language Models (LLMs) have proven powerful, but the risk of privacy leakage remains a significant concern. Traditional privacy-preserving methods, such as Differential Privacy and Homomorphic Encryption, are inadequate for black-box API-only settings, demanding either model transparency or heavy computational resources. We propose Prompt2Forget (P2F), the first framework designed to tackle the LLM local privacy challenge by teaching LLM to forget. The method involves decomposing full questions into smaller segments, generating fabricated answers, and obfuscating the model's memory of the original input. A benchmark dataset was crafted with questions containing privacy-sensitive information from diverse fields. P2F achieves zero-shot generalization, allowing adaptability across a wide range of use cases without manual adjustments. Experimental results indicate P2F's robust capability to obfuscate LLM's memory, attaining a forgetfulness score of around 90\% without any utility los
    

