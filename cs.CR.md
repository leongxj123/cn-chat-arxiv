# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Gradient Langevin Unlearning](https://arxiv.org/abs/2403.17105) | 本工作提出了随机梯度 Langevin 反遗忘方法，为近似反遗忘问题提供了隐私保障，并展示了小批次梯度更新相较于全批次的优越性能。 |
| [^2] | [Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models](https://arxiv.org/abs/2402.06659) | Shadowcast是一种隐秘的数据污染攻击方法，可以通过伪装成良性图像和匹配文本来操纵视觉语言模型的响应。它包括标签攻击和说服攻击，可以混淆类别标签并编写有说服力的描述。使用仅50个毒样本，Shadowcast能够高效实现攻击者的意图。 |
| [^3] | [Flamingo: Multi-Round Single-Server Secure Aggregation with Applications to Private Federated Learning.](http://arxiv.org/abs/2308.09883) | Flamingo是一个用于实现跨大量客户端安全聚合的系统，在私有联邦学习中有广泛的应用。通过消除每轮设置和引入轻量级的丢失容忍协议，Flamingo解决了以往协议在多轮设置下的问题，并引入了新的本地选择客户端邻域的方式。 |
| [^4] | [Training Data Protection with Compositional Diffusion Models.](http://arxiv.org/abs/2308.01937) | 使用分区扩散模型（CDM）训练不同的扩散模型，并在推断时任意组合它们，实现了训练数据保护和选择性遗忘，同时还可以根据用户访问权限提供定制模型。 |

# 详细

[^1]: 随机梯度 Langevin 反遗忘

    Stochastic Gradient Langevin Unlearning

    [https://arxiv.org/abs/2403.17105](https://arxiv.org/abs/2403.17105)

    本工作提出了随机梯度 Langevin 反遗忘方法，为近似反遗忘问题提供了隐私保障，并展示了小批次梯度更新相较于全批次的优越性能。

    

    “被遗忘的权利”是用户数据隐私的法律所确保的越来越重要。机器反遗忘旨在高效地消除已训练模型参数上某些数据点的影响，使其近似于从头开始重新训练模型。本研究提出了随机梯度 Langevin 反遗忘，这是第一个基于带有隐私保障的噪声随机梯度下降（SGD）的反遗忘框架，适用于凸性假设下的近似反遗忘问题。我们的结果表明，与全批次对应方法相比，小批次梯度更新在隐私复杂度权衡方面提供了更好的性能。我们的反遗忘方法具有诸多算法优势，包括与重新训练相比的复杂度节省，以及支持顺序和批量反遗忘。为了检验我们方法的隐私-效用-复杂度权衡，我们在基准数据集上进行了实验比较。

    arXiv:2403.17105v1 Announce Type: new  Abstract: ``The right to be forgotten'' ensured by laws for user data privacy becomes increasingly important. Machine unlearning aims to efficiently remove the effect of certain data points on the trained model parameters so that it can be approximately the same as if one retrains the model from scratch. This work proposes stochastic gradient Langevin unlearning, the first unlearning framework based on noisy stochastic gradient descent (SGD) with privacy guarantees for approximate unlearning problems under convexity assumption. Our results show that mini-batch gradient updates provide a superior privacy-complexity trade-off compared to the full-batch counterpart. There are numerous algorithmic benefits of our unlearning approach, including complexity saving compared to retraining, and supporting sequential and batch unlearning. To examine the privacy-utility-complexity trade-off of our method, we conduct experiments on benchmark datasets compared 
    
[^2]: Shadowcast: 隐秘的数据污染攻击对抗视觉语言模型

    Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models

    [https://arxiv.org/abs/2402.06659](https://arxiv.org/abs/2402.06659)

    Shadowcast是一种隐秘的数据污染攻击方法，可以通过伪装成良性图像和匹配文本来操纵视觉语言模型的响应。它包括标签攻击和说服攻击，可以混淆类别标签并编写有说服力的描述。使用仅50个毒样本，Shadowcast能够高效实现攻击者的意图。

    

    视觉语言模型（VLM）能够从视觉输入中生成文本响应，然而它们的多功能性带来了重大的安全隐患。本研究首次揭示了VLM对数据污染攻击的易受性，这些攻击可以操纵对无害的日常提示的响应。我们引入了一种名为Shadowcast的隐秘数据污染攻击方法，其中毒样本在视觉上与具有匹配文本的良性图像难以区分。Shadowcast在两种攻击类型中展示出了有效性。第一种是标签攻击，使VLM误识别类别标签，例如混淆唐纳德·特朗普和乔·拜登等人。第二种是说服攻击，利用VLM的文本生成能力来编写故事，例如通过有说服力和看似合理的描述将垃圾食品描绘成健康食品。我们展示了Shadowcast使用仅50个毒样本就能高度有效地实现攻击者的意图。此外，这些毒样本仍然保持有效。

    Vision-Language Models (VLMs) excel in generating textual responses from visual inputs, yet their versatility raises significant security concerns. This study takes the first step in exposing VLMs' susceptibility to data poisoning attacks that can manipulate responses to innocuous, everyday prompts. We introduce Shadowcast, a stealthy data poisoning attack method where poison samples are visually indistinguishable from benign images with matching texts. Shadowcast demonstrates effectiveness in two attack types. The first is Label Attack, tricking VLMs into misidentifying class labels, such as confusing Donald Trump for Joe Biden. The second is Persuasion Attack, which leverages VLMs' text generation capabilities to craft narratives, such as portraying junk food as health food, through persuasive and seemingly rational descriptions. We show that Shadowcast are highly effective in achieving attacker's intentions using as few as 50 poison samples. Moreover, these poison samples remain eff
    
[^3]: Flamingo: 多轮单服务器安全聚合及其在私有联邦学习中的应用

    Flamingo: Multi-Round Single-Server Secure Aggregation with Applications to Private Federated Learning. (arXiv:2308.09883v1 [cs.CR])

    [http://arxiv.org/abs/2308.09883](http://arxiv.org/abs/2308.09883)

    Flamingo是一个用于实现跨大量客户端安全聚合的系统，在私有联邦学习中有广泛的应用。通过消除每轮设置和引入轻量级的丢失容忍协议，Flamingo解决了以往协议在多轮设置下的问题，并引入了新的本地选择客户端邻域的方式。

    

    本文介绍了Flamingo，这是一个用于跨大量客户端安全聚合数据的系统。在安全聚合中，服务器对客户端的私有输入进行求和，并在不了解个体输入的情况下得到结果，仅能推断出最终总和。Flamingo专注于联邦学习中的多轮设置，其中执行多个连续的模型权重求和（平均），以得到一个良好的模型。之前的协议（例如Bell等人的CCS '20）仅适用于单轮，并通过多次重复该协议来适应联邦学习的设置。Flamingo消除了之前协议每轮设置的需求，并引入了一种新的轻量级的丢失容忍协议，以确保如果客户端在求和过程中离开，服务器仍然可以获得有意义的结果。此外，Flamingo还引入了一种新的本地选择所谓的客户端邻域的方式，此概念由Bell等人提出。

    This paper introduces Flamingo, a system for secure aggregation of data across a large set of clients. In secure aggregation, a server sums up the private inputs of clients and obtains the result without learning anything about the individual inputs beyond what is implied by the final sum. Flamingo focuses on the multi-round setting found in federated learning in which many consecutive summations (averages) of model weights are performed to derive a good model. Previous protocols, such as Bell et al. (CCS '20), have been designed for a single round and are adapted to the federated learning setting by repeating the protocol multiple times. Flamingo eliminates the need for the per-round setup of previous protocols, and has a new lightweight dropout resilience protocol to ensure that if clients leave in the middle of a sum the server can still obtain a meaningful result. Furthermore, Flamingo introduces a new way to locally choose the so-called client neighborhood introduced by Bell et al
    
[^4]: 使用组合扩散模型实现训练数据保护

    Training Data Protection with Compositional Diffusion Models. (arXiv:2308.01937v1 [cs.LG])

    [http://arxiv.org/abs/2308.01937](http://arxiv.org/abs/2308.01937)

    使用分区扩散模型（CDM）训练不同的扩散模型，并在推断时任意组合它们，实现了训练数据保护和选择性遗忘，同时还可以根据用户访问权限提供定制模型。

    

    我们引入了分区扩散模型（CDM），一种在不同数据源上训练不同扩散模型（或提示）并在推断时任意组合它们的方法。这些单独的模型可以在孤立状态下、在不同时间、在不同分布和领域上进行训练，并可以后续组合以达到与同时训练所有数据的理想模型相当的性能。此外，每个模型只包含其在训练期间接触到的数据子集的信息，可以实现多种形式的训练数据保护。特别是，CDM是第一种可以实现大规模扩散模型的选择性遗忘和持续学习的方法，并且允许根据用户访问权限提供定制模型。CDM还可以确定生成特定样本的数据子集的重要性。

    We introduce Compartmentalized Diffusion Models (CDM), a method to train different diffusion models (or prompts) on distinct data sources and arbitrarily compose them at inference time. The individual models can be trained in isolation, at different times, and on different distributions and domains and can be later composed to achieve performance comparable to a paragon model trained on all data simultaneously. Furthermore, each model only contains information about the subset of the data it was exposed to during training, enabling several forms of training data protection. In particular, CDMs are the first method to enable both selective forgetting and continual learning for large-scale diffusion models, as well as allowing serving customized models based on the user's access rights. CDMs also allow determining the importance of a subset of the data in generating particular samples.
    

