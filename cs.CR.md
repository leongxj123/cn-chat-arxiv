# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Preserving Prompt Tuning for Large Language Model Services.](http://arxiv.org/abs/2305.06212) | RAPT是一个提供隐私保证的大语言模型服务的提示调整框架，采用本地差分隐私设置和新颖的隐私化标记重建任务，并在多种任务中取得有竞争力的性能和良好的隐私保护效果。 |
| [^2] | [Privacy-Preserving CNN Training with Transfer Learning.](http://arxiv.org/abs/2304.03807) | 本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。 |

# 详细

[^1]: 大语言模型服务的隐私保护提示调整

    Privacy-Preserving Prompt Tuning for Large Language Model Services. (arXiv:2305.06212v1 [cs.CL])

    [http://arxiv.org/abs/2305.06212](http://arxiv.org/abs/2305.06212)

    RAPT是一个提供隐私保证的大语言模型服务的提示调整框架，采用本地差分隐私设置和新颖的隐私化标记重建任务，并在多种任务中取得有竞争力的性能和良好的隐私保护效果。

    

    提示调整为用户在新兴的大语言模型服务场景下使用其私有数据自定义大语言模型(LLM)的有效方式。但是，私有数据的敏感性需要在LLM服务定制中保护隐私。基于提示调整，我们提出了一种名为隐私保护提示调整(RAPT)的框架，为LLM服务提供隐私保证。RAPT采用本地隐私设置，允许用户使用本地差分隐私对其数据进行本地化隐私处理。由于在直接训练隐私化数据的情况下，提示调整表现不佳，因此我们引入了一种新颖的隐私化标记重建任务，与下游任务一起进行培训，使LLM学习更好的任务相关表示。尽管我们的框架简单，但实验表明，RAPT在各种任务中均具有竞争力的性能，并提供抵御对手的隐私保证。

    Prompt tuning provides an efficient way for users to customize Large Language Models (LLMs) with their private data in the emerging LLM service scenario. However, the sensitive nature of private data brings the need for privacy preservation in LLM service customization. Based on prompt tuning, we propose Privacy-Preserving Prompt Tuning (RAPT), a framework that provides privacy guarantees for LLM services. \textsc{rapt} adopts a local privacy setting, allowing users to privatize their data locally with local differential privacy. As prompt tuning performs poorly when directly trained on privatized data, we introduce a novel privatized token reconstruction task that is trained jointly with the downstream task, allowing LLMs to learn better task-dependent representations. Despite the simplicity of our framework, experiments show that RAPT achieves competitive performance across tasks while providing privacy guarantees against adversaries.
    
[^2]: 使用迁移学习实现隐私保护的CNN训练

    Privacy-Preserving CNN Training with Transfer Learning. (arXiv:2304.03807v1 [cs.CR])

    [http://arxiv.org/abs/2304.03807](http://arxiv.org/abs/2304.03807)

    本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。

    

    隐私保护的神经网络推理已经得到很好的研究，同时保持同态CNN训练仍然是一项挑战性的任务。在本文中，我们提出了一种实用的解决方案来实现基于同态加密技术的隐私保护CNN训练。据我们所知，这是第一次成功突破这个难题，以前没有任何工作达到这个目标。采用了几种技术：（1）通过迁移学习，可以将隐私保护的CNN训练简化为同态神经网络训练，甚至是多类逻辑回归（MLR）训练；（2）通过更快的梯度变体$\texttt{Quadratic Gradient}$，应用于MLR的增强梯度方法，在收敛速度方面具有最先进的性能；（3）我们采用数学中的变换思想，将加密域中的近似Softmax函数转换成已经研究过的逼近方法，从而得到更好的结果。

    Privacy-preserving nerual network inference has been well studied while homomorphic CNN training still remains an open challenging task. In this paper, we present a practical solution to implement privacy-preserving CNN training based on mere Homomorphic Encryption (HE) technique. To our best knowledge, this is the first attempt successfully to crack this nut and no work ever before has achieved this goal. Several techniques combine to make it done: (1) with transfer learning, privacy-preserving CNN training can be reduced to homomorphic neural network training, or even multiclass logistic regression (MLR) training; (2) via a faster gradient variant called $\texttt{Quadratic Gradient}$, an enhanced gradient method for MLR with a state-of-the-art performance in converge speed is applied in this work to achieve high performance; (3) we employ the thought of transformation in mathematics to transform approximating Softmax function in encryption domain to the well-studied approximation of 
    

