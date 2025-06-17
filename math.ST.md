# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sample complexity of quantum hypothesis testing](https://arxiv.org/abs/2403.17868) | 本文研究了量子假设检验的样本复杂度，得出了对称和非对称设置中的二进制量子假设检验的样本复杂度与反错误概率的对数和保真度的负对数的关系。 |
| [^2] | [Conformal Risk Control.](http://arxiv.org/abs/2208.02814) | 该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。 |

# 详细

[^1]: 量子假设检验的样本复杂度

    Sample complexity of quantum hypothesis testing

    [https://arxiv.org/abs/2403.17868](https://arxiv.org/abs/2403.17868)

    本文研究了量子假设检验的样本复杂度，得出了对称和非对称设置中的二进制量子假设检验的样本复杂度与反错误概率的对数和保真度的负对数的关系。

    

    传统上，人们从信息论的角度研究量子假设检验，在这种情况下，人们对错误概率的最优衰减速率感兴趣，这个速率是未知状态的样本数量函数。本文研究了量子假设检验的样本复杂度，旨在确定达到所需错误概率所需的最少样本数量。通过利用已有文献中关于量子假设检验的丰富知识，我们表征了对称和非对称设置中的二进制量子假设检验的样本复杂度，并提供了多个量子假设检验的样本复杂度的界限。更详细地说，我们证明了对称二进制量子假设检验的样本复杂度对反错误概率的对数和保真度的负对数的对数。

    arXiv:2403.17868v1 Announce Type: cross  Abstract: Quantum hypothesis testing has been traditionally studied from the information-theoretic perspective, wherein one is interested in the optimal decay rate of error probabilities as a function of the number of samples of an unknown state. In this paper, we study the sample complexity of quantum hypothesis testing, wherein the goal is to determine the minimum number of samples needed to reach a desired error probability. By making use of the wealth of knowledge that already exists in the literature on quantum hypothesis testing, we characterize the sample complexity of binary quantum hypothesis testing in the symmetric and asymmetric settings, and we provide bounds on the sample complexity of multiple quantum hypothesis testing. In more detail, we prove that the sample complexity of symmetric binary quantum hypothesis testing depends logarithmically on the inverse error probability and inversely on the negative logarithm of the fidelity. 
    
[^2]: 一种符合保序的风险控制方法

    Conformal Risk Control. (arXiv:2208.02814v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.02814](http://arxiv.org/abs/2208.02814)

    该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。

    

    我们将符合性预测推广至控制任何单调损失函数的期望值。该算法将分裂符合性预测及其覆盖保证进行了泛化。类似于符合性预测，符合保序的风险控制方法在$\mathcal{O}(1/n)$因子内保持紧密性。计算机视觉和自然语言处理领域的示例证明了我们算法在控制误报率、图形距离和令牌级F1得分方面的应用。

    We extend conformal prediction to control the expected value of any monotone loss function. The algorithm generalizes split conformal prediction together with its coverage guarantee. Like conformal prediction, the conformal risk control procedure is tight up to an $\mathcal{O}(1/n)$ factor. Worked examples from computer vision and natural language processing demonstrate the usage of our algorithm to bound the false negative rate, graph distance, and token-level F1-score.
    

