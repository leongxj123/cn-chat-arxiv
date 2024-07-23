# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples](https://arxiv.org/abs/2403.05181) | 本文提出了一种训练教师模型的方法，通过引入敌对示例的稀疏输出，并与标准训练数据结合使用，来加强教师模型对学生蒸馏的防御。 |
| [^2] | [Advancing TTP Analysis: Harnessing the Power of Encoder-Only and Decoder-Only Language Models with Retrieval Augmented Generation.](http://arxiv.org/abs/2401.00280) | 本研究探索了如何利用编码器模型和解码器模型来理解和总结网络攻击过程中的策略和目的，使用检索增强生成技术来提取相关上下文，并解决了现有模型在网络安全领域中产生错误信息的问题。 |
| [^3] | [Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing.](http://arxiv.org/abs/2301.12554) | 本文研究通过混合标准分类器和鲁棒模型的输出概率来减轻准确性和鲁棒性之间的权衡问题，进而提高分类器的鲁棒性。同时提出了一种自适应平滑的方法，可以降低实现鲁棒性的准确度惩罚。 |
| [^4] | [Does Federated Learning Really Need Backpropagation?.](http://arxiv.org/abs/2301.12195) | 本文提出一种不需要反向传播的联邦学习框架BAFFLE，该框架使用多个正向过程估计梯度，具有高内存效率，容易适应上传带宽，与硬件优化和模型量化/修剪兼容，适用于受信任的执行环境。 |

# 详细

[^1]: Adversarial Sparse Teacher: 对抗敌对示例，防御用对抗示例进行的基于蒸馏的模型窃取攻击

    Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples

    [https://arxiv.org/abs/2403.05181](https://arxiv.org/abs/2403.05181)

    本文提出了一种训练教师模型的方法，通过引入敌对示例的稀疏输出，并与标准训练数据结合使用，来加强教师模型对学生蒸馏的防御。

    

    知识蒸馏（KD）促进了将高级教师模型的区分能力转移到更简单的学生模型，确保提高性能而不影响准确性。它也被用于模型窃取攻击，其中对手使用KD来模仿教师模型的功能。最近在该领域的发展受到了吝啬教师模型的影响，该模型通过实证分析表明稀疏输出可以显著降低学生模型的性能。为了解决知识产权泄露的风险，我们的工作引入了一种训练教师模型的方法，该方法从根本上保护其logits，受“恶毒教师”理念的影响。与现有方法不同，我们将对抗示例的稀疏输出与标准训练数据结合起来，以加强教师对学生蒸馏的防御。我们的方法巧妙地减少了相对的e

    arXiv:2403.05181v1 Announce Type: new  Abstract: Knowledge Distillation (KD) facilitates the transfer of discriminative capabilities from an advanced teacher model to a simpler student model, ensuring performance enhancement without compromising accuracy. It is also exploited for model stealing attacks, where adversaries use KD to mimic the functionality of a teacher model. Recent developments in this domain have been influenced by the Stingy Teacher model, which provided empirical analysis showing that sparse outputs can significantly degrade the performance of student models. Addressing the risk of intellectual property leakage, our work introduces an approach to train a teacher model that inherently protects its logits, influenced by the Nasty Teacher concept. Differing from existing methods, we incorporate sparse outputs of adversarial examples with standard training data to strengthen the teacher's defense against student distillation. Our approach carefully reduces the relative e
    
[^2]: 推进TTP分析：利用仅编码器和仅解码器语言模型并提升的生成模型的力量

    Advancing TTP Analysis: Harnessing the Power of Encoder-Only and Decoder-Only Language Models with Retrieval Augmented Generation. (arXiv:2401.00280v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2401.00280](http://arxiv.org/abs/2401.00280)

    本研究探索了如何利用编码器模型和解码器模型来理解和总结网络攻击过程中的策略和目的，使用检索增强生成技术来提取相关上下文，并解决了现有模型在网络安全领域中产生错误信息的问题。

    

    战术，技术和程序（TTPs）概述了攻击者利用漏洞的方法。由于假定的专业知识，复杂的依赖关系和内在的模糊性，对MITRE ATT＆CK框架中的TTPs的解释对于网络安全从业人员来说可能具有挑战性。与此同时，大型语言模型（LLMs）的进步导致了最近在研究中探索其在网络安全操作中的用途的激增。这引起了我们的疑问，仅编码器（例如RoBERTa）和仅解码器（例如GPT-3.5）LLMs对于理解和总结TTPs以通知分析人员有关网络攻击过程的预期目的（即策略）的能力如何。最先进的LLMs已经显示出容易产生错误信息，这在网络安全等关键领域是有问题的。因此，我们提出使用检索增强生成（RAG）技术来为仅解码器的LLMs提取每个网络攻击过程的相关上下文（无需微调）。

    Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise, complex dependencies, and inherent ambiguity. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. This leads us to question how well encoder-only (e.g., RoBERTa) and decoder-only (e.g., GPT-3.5) LLMs can comprehend and summarize TTPs to inform analysts of the intended purposes (i.e., tactics) of a cyberattack procedure. The state-of-the-art LLMs have shown to be prone to hallucination by providing inaccurate information, which is problematic in critical domains like cybersecurity. Therefore, we propose the use of Retrieval Augmented Generation (RAG) techniques to extract relevant contexts for each cyberattack procedure for decoder-only LLMs (without fine-tuning)
    
[^3]: 通过自适应平滑改善分类器的准确性-鲁棒性平衡

    Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing. (arXiv:2301.12554v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12554](http://arxiv.org/abs/2301.12554)

    本文研究通过混合标准分类器和鲁棒模型的输出概率来减轻准确性和鲁棒性之间的权衡问题，进而提高分类器的鲁棒性。同时提出了一种自适应平滑的方法，可以降低实现鲁棒性的准确度惩罚。

    

    尽管以前的研究提出了大量增强神经分类器对抗鲁棒性的方法，但由于在清晰度方面存在不可接受的严重惩罚，实践者仍然不愿采用这些技术。本文表明，通过混合标准分类器和强鲁棒模型的输出概率，其中标准网络优化清晰度而不是一般的鲁棒性，可以显着减轻这种准确性-鲁棒性权衡问题。我们显示出基于鲁棒性的基本分类器的正确和不正确示例的置信度差异是这种改善的关键因素。除提供直观和经验证据外，我们还在现实假设下理论上证明了混合分类器的鲁棒性。此外，我们还将一个对抗性输入检测器适应为混合网络，自适应地调整两个基本模型的混合，从而进一步减少实现鲁棒性的准确性惩罚。

    While prior research has proposed a plethora of methods that enhance the adversarial robustness of neural classifiers, practitioners are still reluctant to adopt these techniques due to their unacceptably severe penalties in clean accuracy. This paper shows that by mixing the output probabilities of a standard classifier and a robust model, where the standard network is optimized for clean accuracy and is not robust in general, this accuracy-robustness trade-off can be significantly alleviated. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key ingredient of this improvement. In addition to providing intuitive and empirical evidence, we also theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The 
    
[^4]: 《联邦学习是否真正需要反向传播？》

    Does Federated Learning Really Need Backpropagation?. (arXiv:2301.12195v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12195](http://arxiv.org/abs/2301.12195)

    本文提出一种不需要反向传播的联邦学习框架BAFFLE，该框架使用多个正向过程估计梯度，具有高内存效率，容易适应上传带宽，与硬件优化和模型量化/修剪兼容，适用于受信任的执行环境。

    

    联邦学习（FL）是一种去中心化地让客户端共同训练一个服务器模型的一般性原则，而无需共享本地数据。FL是一个具有实际应用的有前途的框架，但其标准训练范式要求客户端通过模型进行反向传播以计算梯度。由于这些客户端通常是边缘设备而不是完全受信任的，因此在它们上执行反向传播会产生计算和存储开销以及白盒漏洞。因此，我们开发了一种不需要反向传播的联邦学习，称为BAFFLE，其中反向传播替换为多个正向过程以估计梯度。BAFFLE具有以下优点：1）内存效率高并且容易适应上传带宽；2）与仅推理硬件优化以及模型量化或修剪兼容；3）非常适合受信任的执行环境，因为BAFFLE中的客户端仅执行正向传播并返回一组标量到服务器。我们通过实验使用了BAFFLE的优越性能。

    Federated learning (FL) is a general principle for decentralized clients to train a server model collectively without sharing local data. FL is a promising framework with practical applications, but its standard training paradigm requires the clients to backpropagate through the model to compute gradients. Since these clients are typically edge devices and not fully trusted, executing backpropagation on them incurs computational and storage overhead as well as white-box vulnerability. In light of this, we develop backpropagation-free federated learning, dubbed BAFFLE, in which backpropagation is replaced by multiple forward processes to estimate gradients. BAFFLE is 1) memory-efficient and easily fits uploading bandwidth; 2) compatible with inference-only hardware optimization and model quantization or pruning; and 3) well-suited to trusted execution environments, because the clients in BAFFLE only execute forward propagation and return a set of scalars to the server. Empirically we us
    

