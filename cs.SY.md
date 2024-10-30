# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Blue and Green-Mode Energy-Efficient Chemiresistive Sensor Array Realized by Rapid Ensemble Learning](https://arxiv.org/abs/2403.01642) | 该研究提出了一种通过快速集成学习优化化学传感器阵列的策略，引入了蓝色和绿色两种工作模式，通过选择性激活关键传感器显著降低能耗而不影响检测准确性。 |
| [^2] | [C-GAIL: Stabilizing Generative Adversarial Imitation Learning with Control Theory](https://arxiv.org/abs/2402.16349) | 该论文利用控制理论改进了生成对抗模仿学习（GAIL），提出了一种名为“Controlled-GAIL”（C-GAIL）的算法，能够解决GAIL训练不稳定性的问题，并在MuJoCo任务中取得了较快的收敛速度。 |

# 详细

[^1]: 通过快速集成学习实现的蓝绿模式高能效化化学传感器阵列

    Blue and Green-Mode Energy-Efficient Chemiresistive Sensor Array Realized by Rapid Ensemble Learning

    [https://arxiv.org/abs/2403.01642](https://arxiv.org/abs/2403.01642)

    该研究提出了一种通过快速集成学习优化化学传感器阵列的策略，引入了蓝色和绿色两种工作模式，通过选择性激活关键传感器显著降低能耗而不影响检测准确性。

    

    arXiv:2403.01642v1 公告类型: 新的 摘要: 物联网的快速发展需要开发既高效又能胜任的优化化学传感器阵列。本研究引入一种新颖的优化策略，采用快速集成学习模型委员会方法来实现这些目标。利用弹性网回归、随机森林、XGBoost等机器学习模型，该策略识别出在化学传感器阵列中对准确分类最具影响力的传感器：引入加权投票机制来聚合传感器选择中的模型意见，从而建立了两种不同的工作模式，称为“蓝色”和“绿色”。蓝色模式利用所有传感器进行最大检测能力，而绿色模式仅选择性激活关键传感器，从而显著降低能耗而不影响检测准确性。该策略通过理论验证。

    arXiv:2403.01642v1 Announce Type: new  Abstract: The rapid advancement of Internet of Things (IoT) necessitates the development of optimized Chemiresistive Sensor (CRS) arrays that are both energy-efficient and capable. This study introduces a novel optimization strategy that employs a rapid ensemble learning-based model committee approach to achieve these goals. Utilizing machine learning models such as Elastic Net Regression, Random Forests, and XGBoost, among others, the strategy identifies the most impactful sensors in a CRS array for accurate classification: A weighted voting mechanism is introduced to aggregate the models' opinions in sensor selection, thereby setting up wo distinct working modes, termed "Blue" and "Green". The Blue mode operates with all sensors for maximum detection capability, while the Green mode selectively activates only key sensors, significantly reducing energy consumption without compromising detection accuracy. The strategy is validated through theoreti
    
[^2]: C-GAIL: 利用控制理论稳定生成对抗模仿学习

    C-GAIL: Stabilizing Generative Adversarial Imitation Learning with Control Theory

    [https://arxiv.org/abs/2402.16349](https://arxiv.org/abs/2402.16349)

    该论文利用控制理论改进了生成对抗模仿学习（GAIL），提出了一种名为“Controlled-GAIL”（C-GAIL）的算法，能够解决GAIL训练不稳定性的问题，并在MuJoCo任务中取得了较快的收敛速度。

    

    生成对抗模仿学习（GAIL）训练一个生成策略来模仿一个演示者。它使用基于策略的强化学习（RL）来优化从类似GAN的鉴别器中导出的奖励信号。GAIL的一个主要缺点是其训练不稳定性 - 它继承了GAN的复杂训练动态，以及RL引入的分布转移。这可能导致训练过程中的振荡，从而影响其样本效率和最终策略性能。最近的工作表明，控制理论可以帮助GAN的训练收敛。本文延伸了这一线路的工作，对GAIL进行了控制理论分析，并导出了一种新颖的控制器，该控制器不仅将GAIL推向期望的均衡点，还在“单步”设置中实现了渐近稳定性。基于此，我们提出了一个实用算法“Controlled-GAIL”（C-GAIL）。在MuJoCo任务中，我们的受控变体能够加速收敛速度。

    arXiv:2402.16349v1 Announce Type: new  Abstract: Generative Adversarial Imitation Learning (GAIL) trains a generative policy to mimic a demonstrator. It uses on-policy Reinforcement Learning (RL) to optimize a reward signal derived from a GAN-like discriminator. A major drawback of GAIL is its training instability - it inherits the complex training dynamics of GANs, and the distribution shift introduced by RL. This can cause oscillations during training, harming its sample efficiency and final policy performance. Recent work has shown that control theory can help with the convergence of a GAN's training. This paper extends this line of work, conducting a control-theoretic analysis of GAIL and deriving a novel controller that not only pushes GAIL to the desired equilibrium but also achieves asymptotic stability in a 'one-step' setting. Based on this, we propose a practical algorithm 'Controlled-GAIL' (C-GAIL). On MuJoCo tasks, our controlled variant is able to speed up the rate of conve
    

