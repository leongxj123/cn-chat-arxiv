# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Whispers in the Machine: Confidentiality in LLM-integrated Systems](https://arxiv.org/abs/2402.06922) | 本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。 |

# 详细

[^1]: 机器中的私语：LLM集成系统中的保密性

    Whispers in the Machine: Confidentiality in LLM-integrated Systems

    [https://arxiv.org/abs/2402.06922](https://arxiv.org/abs/2402.06922)

    本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。

    

    大规模语言模型（LLM）越来越多地与外部工具集成。尽管这些集成可以显著提高LLM的功能，但它们也在不同组件之间创建了一个新的攻击面，可能泄露机密数据。具体而言，恶意工具可以利用LLM本身的漏洞来操纵模型并损害其他服务的数据，这引发了在LLM集成环境中如何保护私密数据的问题。在这项工作中，我们提供了一种系统评估LLM集成系统保密性的方法。为此，我们形式化了一个"秘密密钥"游戏，可以捕捉模型隐藏私人信息的能力。这使我们能够比较模型对保密性攻击的脆弱性以及不同防御策略的有效性。在这个框架中，我们评估了八种先前发表的攻击和四种防御方法。我们发现当前的防御方法缺乏泛化性能。

    Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.   In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization
    

