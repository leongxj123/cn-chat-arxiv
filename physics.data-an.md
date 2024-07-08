# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mixed Noise and Posterior Estimation with Conditional DeepGEM](https://arxiv.org/abs/2402.02964) | 本论文提出了一种用于联合估计贝叶斯逆问题中后验概率和噪声参数的新算法，该算法通过期望最大化（EM）算法解决问题，并使用条件标准化流来近似后验概率。该模型能够整合来自多个测量的信息。 |

# 详细

[^1]: 混合噪声与条件深度生成模型下的后验估计

    Mixed Noise and Posterior Estimation with Conditional DeepGEM

    [https://arxiv.org/abs/2402.02964](https://arxiv.org/abs/2402.02964)

    本论文提出了一种用于联合估计贝叶斯逆问题中后验概率和噪声参数的新算法，该算法通过期望最大化（EM）算法解决问题，并使用条件标准化流来近似后验概率。该模型能够整合来自多个测量的信息。

    

    受混合噪声模型的间接测量和纳米计量应用的启发，我们开发了一种新的算法，用于联合估计贝叶斯逆问题中的后验概率和噪声参数。我们提出通过期望最大化（EM）算法来解决这个问题。基于当前的噪声参数，我们在E步中学习了一个条件标准化流，以近似后验概率。在M步中，我们提出再次通过EM算法找到噪声参数的更新，其具有解析公式。我们将条件标准化流的训练与前向和反向KL进行比较，并展示我们的模型能够整合来自许多测量的信息，而不像之前的方法。

    Motivated by indirect measurements and applications from nanometrology with a mixed noise model, we develop a novel algorithm for jointly estimating the posterior and the noise parameters in Bayesian inverse problems. We propose to solve the problem by an expectation maximization (EM) algorithm. Based on the current noise parameters, we learn in the E-step a conditional normalizing flow that approximates the posterior. In the M-step, we propose to find the noise parameter updates again by an EM algorithm, which has analytical formulas. We compare the training of the conditional normalizing flow with the forward and reverse KL, and show that our model is able to incorporate information from many measurements, unlike previous approaches.
    

