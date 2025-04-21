# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) | 展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。 |
| [^2] | [Energy-Latency Attacks via Sponge Poisoning.](http://arxiv.org/abs/2203.08147) | 本文探讨了一种名为“海绵毒化”的攻击方法，首次证明了在训练时注入海绵样本可以在测试时提高机器学习模型在每个输入上的能耗和延迟，并且即使攻击者只控制了一些模型更新也可以进行此攻击，海绵毒化几乎完全消除了硬件加速器的效果。 |

# 详细

[^1]: 用简单自适应攻击越狱功能对齐的LLM

    Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

    [https://arxiv.org/abs/2404.02151](https://arxiv.org/abs/2404.02151)

    展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。

    

    我们展示了即使是最新的安全对齐的LLM也不具有抵抗简单自适应越狱攻击的稳健性。首先，我们展示了如何成功利用对logprobs的访问进行越狱：我们最初设计了一个对抗性提示模板（有时会适应目标LLM），然后我们在后缀上应用随机搜索以最大化目标logprob（例如token“Sure”），可能会进行多次重启。通过这种方式，我们实现了对GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gemma-7B和针对GCG攻击进行对抗训练的HarmBench上的R2D2等几乎100%的攻击成功率--根据GPT-4的评判。我们还展示了如何通过转移或预填充攻击以100%的成功率对所有不暴露logprobs的Claude模型进行越狱。此外，我们展示了如何在受污染的模型中使用对一组受限制的token执行随机搜索以查找木马字符串的方法--这项任务与许多其他任务共享相同的属性。

    arXiv:2404.02151v1 Announce Type: cross  Abstract: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many s
    
[^2]: 基于海绵毒化的能耗延迟攻击。

    Energy-Latency Attacks via Sponge Poisoning. (arXiv:2203.08147v4 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2203.08147](http://arxiv.org/abs/2203.08147)

    本文探讨了一种名为“海绵毒化”的攻击方法，首次证明了在训练时注入海绵样本可以在测试时提高机器学习模型在每个输入上的能耗和延迟，并且即使攻击者只控制了一些模型更新也可以进行此攻击，海绵毒化几乎完全消除了硬件加速器的效果。

    

    海绵样本是在测试时精心优化的输入，可在硬件加速器上部署时增加神经网络的能量消耗和延迟。本文首次证明了海绵样本也可通过一种名为海绵毒化的攻击注入到训练中。该攻击允许在每个测试时输入中不加区分地提高机器学习模型的能量消耗和延迟。我们提出了一种新的海绵毒化形式化方法，克服了与优化测试时海绵样本相关的限制，并表明即使攻击者仅控制几个模型更新，例如模型训练被外包给不受信任的第三方或通过联邦学习分布式进行，也可以进行这种攻击。我们进行了广泛的实验分析，表明海绵毒化几乎完全消除了硬件加速器的效果。同时，我们还分析了毒化模型的激活，确定了哪些计算对导致能量消耗和延迟增加起重要作用。

    Sponge examples are test-time inputs carefully optimized to increase energy consumption and latency of neural networks when deployed on hardware accelerators. In this work, we are the first to demonstrate that sponge examples can also be injected at training time, via an attack that we call sponge poisoning. This attack allows one to increase the energy consumption and latency of machine-learning models indiscriminately on each test-time input. We present a novel formalization for sponge poisoning, overcoming the limitations related to the optimization of test-time sponge examples, and show that this attack is possible even if the attacker only controls a few model updates; for instance, if model training is outsourced to an untrusted third-party or distributed via federated learning. Our extensive experimental analysis shows that sponge poisoning can almost completely vanish the effect of hardware accelerators. We also analyze the activations of poisoned models, identifying which comp
    

