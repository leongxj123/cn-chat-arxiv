# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UFID: A Unified Framework for Input-level Backdoor Detection on Diffusion Models](https://arxiv.org/abs/2404.01101) | 扩散模型容易受到后门攻击，本文提出了一个统一的框架用于输入级后门检测，弥补了该领域的空白，并不需要访问模型的白盒信息。 |

# 详细

[^1]: UFID: 一个统一的框架用于扩散模型上的输入级后门检测

    UFID: A Unified Framework for Input-level Backdoor Detection on Diffusion Models

    [https://arxiv.org/abs/2404.01101](https://arxiv.org/abs/2404.01101)

    扩散模型容易受到后门攻击，本文提出了一个统一的框架用于输入级后门检测，弥补了该领域的空白，并不需要访问模型的白盒信息。

    

    扩散模型容易受到后门攻击，即恶意攻击者在训练阶段通过对部分训练样本进行毒化来注入后门。为了减轻后门攻击的威胁，对后门检测进行了大量研究。然而，没有人为扩散模型设计了专门的后门检测方法，使得这一领域较少被探索。此外，大多数先前的方法主要集中在传统神经网络的分类任务上，很难轻松地将其适应生成任务上的后门检测。此外，大多数先前的方法需要访问模型权重和架构的白盒访问，或概率logits作为额外信息，这并不总是切实可行的。在本文中

    arXiv:2404.01101v1 Announce Type: cross  Abstract: Diffusion Models are vulnerable to backdoor attacks, where malicious attackers inject backdoors by poisoning some parts of the training samples during the training stage. This poses a serious threat to the downstream users, who query the diffusion models through the API or directly download them from the internet. To mitigate the threat of backdoor attacks, there have been a plethora of investigations on backdoor detections. However, none of them designed a specialized backdoor detection method for diffusion models, rendering the area much under-explored. Moreover, these prior methods mainly focus on the traditional neural networks in the classification task, which cannot be adapted to the backdoor detections on the generative task easily. Additionally, most of the prior methods require white-box access to model weights and architectures, or the probability logits as additional information, which are not always practical. In this paper
    

