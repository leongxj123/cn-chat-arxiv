# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking GPT-4 on Algorithmic Problems: A Systematic Evaluation of Prompting Strategies](https://arxiv.org/abs/2402.17396) | 对GPT-4在算法问题上进行了系统评估，发现采用先进的提示技术可以提高其准确性。 |
| [^2] | [SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks.](http://arxiv.org/abs/2302.13939) | 本论文提出了一种称之为SpikeGPT的生成语言模型，使用二进制、事件驱动脉冲激活单元进行训练，克服了SNN训练中的挑战性。该模型可以用于大规模语言生成任务。 |

# 详细

[^1]: 在算法问题上对GPT-4进行基准测试：关于提示策略的系统评估

    Benchmarking GPT-4 on Algorithmic Problems: A Systematic Evaluation of Prompting Strategies

    [https://arxiv.org/abs/2402.17396](https://arxiv.org/abs/2402.17396)

    对GPT-4在算法问题上进行了系统评估，发现采用先进的提示技术可以提高其准确性。

    

    大型语言模型（LLMs）通过在海量文本语料库上获得的知识在各种下游任务中重新利用，几乎不需要（或根本不需要）调整步骤，从而革新了自然语言处理领域。与此同时，已经反复显示LLMs缺乏系统化泛化，这使得无法将学习到的统计规律外推到训练分布之外。在本研究中，我们对其中一种最先进的LLMs，GPT-4，在三个算法任务上进行了系统基准测试，这些任务通过两个参数控制问题难度。我们比较了GPT-4与其前身（GPT-3.5）以及最近介绍的变压器编码器架构的变体，即神经数据路由器，在解决类似任务时的性能。我们发现采用先进的提示技术可以使GPT-4达到更高的准确性。

    arXiv:2402.17396v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have revolutionized the field of Natural Language Processing thanks to their ability to reuse knowledge acquired on massive text corpora on a wide variety of downstream tasks, with minimal (if any) tuning steps. At the same time, it has been repeatedly shown that LLMs lack systematic generalization, which allows to extrapolate the learned statistical regularities outside the training distribution. In this work, we offer a systematic benchmarking of GPT-4, one of the most advanced LLMs available, on three algorithmic tasks characterized by the possibility to control the problem difficulty with two parameters. We compare the performance of GPT-4 with that of its predecessor (GPT-3.5) and with a variant of the Transformer-Encoder architecture recently introduced to solve similar tasks, the Neural Data Router. We find that the deployment of advanced prompting techniques allows GPT-4 to reach superior accuracy o
    
[^2]: SpikeGPT：带有脉冲神经网络的生成预训练语言模型

    SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks. (arXiv:2302.13939v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.13939](http://arxiv.org/abs/2302.13939)

    本论文提出了一种称之为SpikeGPT的生成语言模型，使用二进制、事件驱动脉冲激活单元进行训练，克服了SNN训练中的挑战性。该模型可以用于大规模语言生成任务。

    

    随着大型语言模型的规模不断扩大，所需的计算资源也随之增加。脉冲神经网络（SNN）已成为一种能够利用稀疏和事件驱动激活减少模型推理计算开销的节能深度学习方法。虽然它们在许多计算机视觉任务上已经具有竞争力，但SNN的训练也被证明更具挑战性。因此，它们的性能落后于现代深度学习，我们尚未看到SNN在语言生成方面的有效性。在本文中，我们受到Receptance Weighted Key Value（RWKV）语言模型的启发，成功实现了“SpikeGPT”，它是一种具有二进制、事件驱动脉冲激活单元的生成语言模型。我们在两种模型变体上训练了所提出的模型：45M和216M参数。据我们所知，SpikeGPT是迄今最大的反向传播训练SNN模型，使其适用于非脉冲模型通常解决的大规模语言生成任务。

    As the size of large language models continue to scale, so does the computational resources required to run it. Spiking Neural Networks (SNNs) have emerged as an energy-efficient approach to deep learning that leverage sparse and event-driven activations to reduce the computational overhead associated with model inference. While they have become competitive with non-spiking models on many computer vision tasks, SNNs have also proven to be more challenging to train. As a result, their performance lags behind modern deep learning, and we are yet to see the effectiveness of SNNs in language generation. In this paper, inspired by the Receptance Weighted Key Value (RWKV) language model, we successfully implement `SpikeGPT', a generative language model with binary, event-driven spiking activation units. We train the proposed model on two model variants: 45M and 216M parameters. To the best of our knowledge, SpikeGPT is the largest backpropagation-trained SNN model to date, rendering it suita
    

