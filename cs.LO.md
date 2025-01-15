# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [End-To-End Set-Based Training for Neural Network Verification.](http://arxiv.org/abs/2401.14961) | 本论文提出了一种端到端基于集合的训练方法，用于训练鲁棒性神经网络进行形式化验证，并证明该方法能够简化验证过程并有效训练出易于验证的神经网络。 |

# 详细

[^1]: 神经网络验证的端到端基于集合的训练方法

    End-To-End Set-Based Training for Neural Network Verification. (arXiv:2401.14961v1 [cs.LG])

    [http://arxiv.org/abs/2401.14961](http://arxiv.org/abs/2401.14961)

    本论文提出了一种端到端基于集合的训练方法，用于训练鲁棒性神经网络进行形式化验证，并证明该方法能够简化验证过程并有效训练出易于验证的神经网络。

    

    神经网络容易受到对抗性攻击，即微小的输入扰动可能导致神经网络输出产生重大变化。安全关键环境需要对输入扰动具有鲁棒性的神经网络。然而，训练和形式化验证鲁棒性神经网络是具有挑战性的。我们首次采用端到端基于集合的训练方法来解决这个挑战，该训练方法能够训练出可进行形式化验证的鲁棒性神经网络。我们的训练方法能够大大简化已训练神经网络的后续形式化鲁棒性验证过程。相比于以往的研究主要关注增强神经网络训练的对抗性攻击，我们的方法利用基于集合的计算来训练整个扰动输入集合上的神经网络。此外，我们证明我们的基于集合的训练方法可以有效训练出易于验证的鲁棒性神经网络。

    Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can result in substantially different outputs of a neural network. Safety-critical environments require neural networks that are robust against input perturbations. However, training and formally verifying robust neural networks is challenging. We address this challenge by employing, for the first time, a end-to-end set-based training procedure that trains robust neural networks for formal verification. Our training procedure drastically simplifies the subsequent formal robustness verification of the trained neural network. While previous research has predominantly focused on augmenting neural network training with adversarial attacks, our approach leverages set-based computing to train neural networks with entire sets of perturbed inputs. Moreover, we demonstrate that our set-based training procedure effectively trains robust neural networks, which are easier to verify. In many cases, set-based trai
    

