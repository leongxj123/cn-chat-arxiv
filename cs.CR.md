# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering.](http://arxiv.org/abs/2310.16152) | 本文提出了一种FLTrojan攻击方法，通过选择性权重篡改，从联邦语言模型中泄露隐私敏感用户数据。通过观察到FL中中间轮次的模型快照可以引起更大的隐私泄露，并发现隐私泄露可以通过篡改模型的选择性权重来加剧。 |

# 详细

[^1]: FLTrojan: 通过选择性权重篡改对联邦语言模型进行隐私泄露攻击

    FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering. (arXiv:2310.16152v1 [cs.CR])

    [http://arxiv.org/abs/2310.16152](http://arxiv.org/abs/2310.16152)

    本文提出了一种FLTrojan攻击方法，通过选择性权重篡改，从联邦语言模型中泄露隐私敏感用户数据。通过观察到FL中中间轮次的模型快照可以引起更大的隐私泄露，并发现隐私泄露可以通过篡改模型的选择性权重来加剧。

    

    联邦学习(Federated learning, FL)正成为许多技术应用中的关键组件，包括语言建模领域，其中个体FL参与者在其本地数据集中往往具有敏感的文本数据。然而，确定联邦语言模型中的隐私泄露程度并不简单，现有的攻击只是试图提取数据，而不考虑数据的敏感性或天真性。为了填补这一空白，在本文中，我们介绍了关于从联邦语言模型中泄露隐私敏感用户数据的两个新发现。首先，我们观察到FL中中间轮次的模型快照比最终训练模型能够造成更大的隐私泄露。其次，我们确定隐私泄露可以通过篡改模型的选择性权重来加剧，这些权重特别负责记忆敏感训练数据。我们展示了恶意客户端如何在FL中泄露其他用户的隐私敏感数据。

    Federated learning (FL) is becoming a key component in many technology-based applications including language modeling -- where individual FL participants often have privacy-sensitive text data in their local datasets. However, realizing the extent of privacy leakage in federated language models is not straightforward and the existing attacks only intend to extract data regardless of how sensitive or naive it is. To fill this gap, in this paper, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other user in FL even
    

