# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Exploration in Bayesian Model-based Reinforcement Learning for Robot Manipulation](https://arxiv.org/abs/2404.01867) | 该论文提出了在机器人操作中基于贝叶斯模型的强化学习中通过主动探索改善模型质量和保持数据效率的方法。 |
| [^2] | [Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series](https://arxiv.org/abs/2402.05203) | 贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。 |
| [^3] | [Energy-Based Models For Speech Synthesis.](http://arxiv.org/abs/2310.12765) | 本文引入了一种新的非自回归模型，即基于能量的模型 (EBMs)，通过噪声对比估计进行训练。该模型可以使用 Langevin MCMC 进行采样，并在语音合成任务中取得了改进。 |

# 详细

[^1]: 机器人操作中基于贝叶斯模型的强化学习中的主动探索

    Active Exploration in Bayesian Model-based Reinforcement Learning for Robot Manipulation

    [https://arxiv.org/abs/2404.01867](https://arxiv.org/abs/2404.01867)

    该论文提出了在机器人操作中基于贝叶斯模型的强化学习中通过主动探索改善模型质量和保持数据效率的方法。

    

    有效解决机器人操作等复杂环境中的多个任务仍然是机器人技术中的一个持续挑战，也是数据驱动解决方案（如强化学习（RL））的一个机会。模型驱动的RL通过构建机器人的动态模型，实现数据重用和在具有相同机器人和类似环境的任务之间进行迁移学习。此外，机器人学中的数据收集成本很高，我们必须依赖数据高效的方法，如基于模型的RL，在基于学习模型的廉价仿真中进行大部分策略学习。因此，模型的质量对于后续任务的性能至关重要。在这项工作中，我们致力于通过在初始探索阶段执行动态模型的主动学习，基于最大化信息收集，以提高模型的质量并保持数据效率。

    arXiv:2404.01867v1 Announce Type: cross  Abstract: Efficiently tackling multiple tasks within complex environment, such as those found in robot manipulation, remains an ongoing challenge in robotics and an opportunity for data-driven solutions, such as reinforcement learning (RL). Model-based RL, by building a dynamic model of the robot, enables data reuse and transfer learning between tasks with the same robot and similar environment. Furthermore, data gathering in robotics is expensive and we must rely on data efficient approaches such as model-based RL, where policy learning is mostly conducted on cheaper simulations based on the learned model. Therefore, the quality of the model is fundamental for the performance of the posterior tasks. In this work, we focus on improving the quality of the model and maintaining the data efficiency by performing active learning of the dynamic model during a preliminary exploration phase based on maximize information gathering. We employ Bayesian ne
    
[^2]: 贝尔曼符合推断：时间序列预测中预测区间的校准

    Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series

    [https://arxiv.org/abs/2402.05203](https://arxiv.org/abs/2402.05203)

    贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。

    

    我们引入了贝尔曼符合推断（BCI），这是一个围绕任何时间序列预测模型的框架，可以提供校准的预测区间。与现有方法不同，BCI能够利用多步预测，并通过在每个时间步骤上解决一维随机控制问题（SCP）来显式优化平均区间长度。特别地，我们使用动态规划算法来找到SCP的最优策略。我们证明了在任意分布转换和时间依赖性下，BCI能够实现长期覆盖，即使多步预测较差。我们在实证中发现，与现有方法相比，BCI避免了无信息区间（长度无限）的生成，并在波动率预测问题上生成了明显更短的预测区间。

    We introduce Bellman Conformal Inference (BCI), a framework that wraps around any time series forecasting models and provides calibrated prediction intervals. Unlike the existing methods, BCI is able to leverage multi-step ahead forecasts and explicitly optimize the average interval lengths by solving a one-dimensional stochastic control problem (SCP) at each time step. In particular, we use the dynamic programming algorithm to find the optimal policy for the SCP. We prove that BCI achieves long-term coverage under arbitrary distribution shifts and temporal dependence, even with poor multi-step ahead forecasts. We find empirically that BCI avoids uninformative intervals that have infinite lengths and generates substantially shorter prediction intervals on volatility forecasting problems when compared with existing methods.
    
[^3]: 基于能量的模型用于语音合成

    Energy-Based Models For Speech Synthesis. (arXiv:2310.12765v1 [cs.SD])

    [http://arxiv.org/abs/2310.12765](http://arxiv.org/abs/2310.12765)

    本文引入了一种新的非自回归模型，即基于能量的模型 (EBMs)，通过噪声对比估计进行训练。该模型可以使用 Langevin MCMC 进行采样，并在语音合成任务中取得了改进。

    

    最近对于非自回归 (non-AR) 模型在语音合成领域引起了很多关注，比如 FastSpeech 2 和扩散模型。与自回归模型不同的是，这些模型的输出之间没有自回归的依赖关系，这使得推断过程更加高效。本文通过引入基于能量的模型 (EBMs) 来拓展非自回归模型的范围。论文描述了如何使用噪声对比估计来训练基于能量的模型，噪声对比估计依赖于正样本和负样本之间的比较。论文提出了一些生成有效负样本的策略，包括使用表现优秀的自回归模型。论文还描述了如何使用 Langevin Markov Chain Monte-Carlo (MCMC) 从基于能量的模型中进行采样。使用 Langevin MCMC 可以建立基于能量的模型与当前流行的扩散模型之间的联系。在 LJSpeech 数据集上的实验证明，所提出的方法相对于 Tacotron 2 有改进。

    Recently there has been a lot of interest in non-autoregressive (non-AR) models for speech synthesis, such as FastSpeech 2 and diffusion models. Unlike AR models, these models do not have autoregressive dependencies among outputs which makes inference efficient. This paper expands the range of available non-AR models with another member called energy-based models (EBMs). The paper describes how noise contrastive estimation, which relies on the comparison between positive and negative samples, can be used to train EBMs. It proposes a number of strategies for generating effective negative samples, including using high-performing AR models. It also describes how sampling from EBMs can be performed using Langevin Markov Chain Monte-Carlo (MCMC). The use of Langevin MCMC enables to draw connections between EBMs and currently popular diffusion models. Experiments on LJSpeech dataset show that the proposed approach offers improvements over Tacotron 2.
    

