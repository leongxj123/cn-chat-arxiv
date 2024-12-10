# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Rounding Implicitly Regularizes Tall-and-Thin Matrices](https://arxiv.org/abs/2403.12278) | 随机舍入技术能有效隐式正则化高瘦矩阵，确保舍入后的矩阵具有完整的列秩。 |
| [^2] | [Revisiting DeepFool: generalization and improvement.](http://arxiv.org/abs/2303.12481) | 本文提出了一种新的对抗性攻击，该攻击是广义了DeepFool攻击，既有效又计算效率高，适用于评估大型深度神经网络的鲁棒性。 |

# 详细

[^1]: 随机舍入隐式正则化高瘦矩阵

    Stochastic Rounding Implicitly Regularizes Tall-and-Thin Matrices

    [https://arxiv.org/abs/2403.12278](https://arxiv.org/abs/2403.12278)

    随机舍入技术能有效隐式正则化高瘦矩阵，确保舍入后的矩阵具有完整的列秩。

    

    受到随机舍入在机器学习和大规模深度神经网络模型训练中的流行，我们考虑实矩阵$\mathbf{A}$的随机近似舍入，其中行数远远多于列数。我们提供了新颖的理论证据，并通过大量实验评估支持，高概率下，随机舍入矩阵的最小奇异值远离零--无论$\mathbf{A}$接近奇异还是$\mathbf{A}$奇异。换句话说，随机舍入\textit{隐式正则化}高瘦矩阵$\mathbf{A}$，使得舍入后的版本具有完整的列秩。我们的证明利用了随机矩阵理论中的有力结果，以及随机舍入误差不集中在低维列空间的思想。

    arXiv:2403.12278v1 Announce Type: new  Abstract: Motivated by the popularity of stochastic rounding in the context of machine learning and the training of large-scale deep neural network models, we consider stochastic nearness rounding of real matrices $\mathbf{A}$ with many more rows than columns. We provide novel theoretical evidence, supported by extensive experimental evaluation that, with high probability, the smallest singular value of a stochastically rounded matrix is well bounded away from zero -- regardless of how close $\mathbf{A}$ is to being rank deficient and even if $\mathbf{A}$ is rank-deficient. In other words, stochastic rounding \textit{implicitly regularizes} tall and skinny matrices $\mathbf{A}$ so that the rounded version has full column rank. Our proofs leverage powerful results in random matrix theory, and the idea that stochastic rounding errors do not concentrate in low-dimensional column spaces.
    
[^2]: 重新审视DeepFool：泛化和改进

    Revisiting DeepFool: generalization and improvement. (arXiv:2303.12481v1 [cs.LG])

    [http://arxiv.org/abs/2303.12481](http://arxiv.org/abs/2303.12481)

    本文提出了一种新的对抗性攻击，该攻击是广义了DeepFool攻击，既有效又计算效率高，适用于评估大型深度神经网络的鲁棒性。

    

    深度神经网络被已知容易受到对抗样本的攻击，这些输入稍加修改便会导致网络做出错误的预测。这导致了大量研究，以评估这些网络对此类扰动的鲁棒性度量。最小l2对抗扰动的鲁棒性，是一种特别重要的鲁棒性度量。然而，现有的用于评估此类鲁棒性度量的方法，要么计算成本高，要么不太准确。在本文中，我们引入了一种新的对抗性攻击方法，它在效果和计算效率之间保持平衡。我们提出的攻击是广义了深度欺骗（DeepFool）攻击，但它们仍然易于理解和实现。我们展示了我们的攻击在效果和计算效率方面均优于现有方法。我们提出的攻击也适用于评估大型深度神经网络的鲁棒性。

    Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal l2 adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large
    

