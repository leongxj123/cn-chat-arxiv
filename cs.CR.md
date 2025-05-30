# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustness-Congruent Adversarial Training for Secure Machine Learning Model Updates](https://arxiv.org/abs/2402.17390) | 通过鲁棒一致对抗训练技术，解决了更新机器学习模型时对抗性鲁棒性和系统安全性的问题。 |
| [^2] | [Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/abs/2402.13459) | 通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。 |

# 详细

[^1]: 针对安全机器学习模型更新的鲁棒一致对抗训练

    Robustness-Congruent Adversarial Training for Secure Machine Learning Model Updates

    [https://arxiv.org/abs/2402.17390](https://arxiv.org/abs/2402.17390)

    通过鲁棒一致对抗训练技术，解决了更新机器学习模型时对抗性鲁棒性和系统安全性的问题。

    

    机器学习模型需要定期更新以提高其平均准确度，利用新颖的架构和额外的数据。然而，新更新的模型可能会犯以前模型未曾犯过的错误。这种误分类被称为负翻转，并被用户体验为性能的退化。在本文中，我们展示了这个问题也影响对抗性样本的鲁棒性，从而阻碍了安全模型更新实践的发展。特别是，当更新模型以提高其对抗性鲁棒性时，一些先前无效的对抗性样本可能会被错误分类，导致系统安全性的认知退化。我们提出了一种名为鲁棒一致对抗训练的新技术来解决这个问题。它涉及使用对抗训练对模型进行微调，同时约束其在对抗性示例上保持更高的鲁棒性。

    arXiv:2402.17390v1 Announce Type: new  Abstract: Machine-learning models demand for periodic updates to improve their average accuracy, exploiting novel architectures and additional data. However, a newly-updated model may commit mistakes that the previous model did not make. Such misclassifications are referred to as negative flips, and experienced by users as a regression of performance. In this work, we show that this problem also affects robustness to adversarial examples, thereby hindering the development of secure model update practices. In particular, when updating a model to improve its adversarial robustness, some previously-ineffective adversarial examples may become misclassified, causing a regression in the perceived security of the system. We propose a novel technique, named robustness-congruent adversarial training, to address this issue. It amounts to fine-tuning a model with adversarial training, while constraining it to retain higher robustness on the adversarial examp
    
[^2]: 学习在指导调优期间操纵大型语言模型

    Learning to Poison Large Language Models During Instruction Tuning

    [https://arxiv.org/abs/2402.13459](https://arxiv.org/abs/2402.13459)

    通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。

    

    大型语言模型（LLMs）的出现标志着语言处理和推理能力方面的重大突破。虽然它们取得了显著进展，但LLMs面临着数据注入攻击的漏洞，其中对手将后门触发器插入训练数据，以操纵输出以进行恶意行为。本研究通过设计一种新的数据注入攻击，旨在利用指导调优过程，进一步识别LLMs中的额外安全风险。我们提出了一种新颖的梯度引导后门触发器学习方法，以有效识别敌对触发器，确保对传统防御手段的规避，同时保持内容的完整性。通过对各种LLMs和任务的实验验证，我们的策略表明在破坏模型输出方面取得了很高的成功率；仅对4,000个指导调优样本中的1％进行注入就导致性能降低率（PDR）约为80％。我们的工作高

    arXiv:2402.13459v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work high
    

