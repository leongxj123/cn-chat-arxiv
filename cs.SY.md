# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment](https://arxiv.org/abs/2402.19085) | 引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。 |
| [^2] | [GPSINDy: Data-Driven Discovery of Equations of Motion.](http://arxiv.org/abs/2309.11076) | GPSINDy是一种数据驱动的方法，通过将高斯过程回归与SINDy相结合，能够从噪声数据中发现非线性动力学系统模型，并在实验证明了其在系统动态和预测未来轨迹方面的改进性能。 |

# 详细

[^1]: 可控偏好优化：朝着可控多目标对齐方向发展

    Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment

    [https://arxiv.org/abs/2402.19085](https://arxiv.org/abs/2402.19085)

    引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。

    

    人工智能中的对齐工作旨在追求模型响应与人类偏好和价值的一致性。本文引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。实验分析表明，经过对齐的模型可以提供符合各种偏好的响应。

    arXiv:2402.19085v1 Announce Type: new  Abstract: Alignment in artificial intelligence pursues the consistency between model responses and human preferences as well as values. In practice, the multifaceted nature of human preferences inadvertently introduces what is known as the "alignment tax" -a compromise where enhancements in alignment within one objective (e.g.,harmlessness) can diminish performance in others (e.g.,helpfulness). However, existing alignment techniques are mostly unidirectional, leading to suboptimal trade-offs and poor flexibility over various objectives. To navigate this challenge, we argue the prominence of grounding LLMs with evident preferences. We introduce controllable preference optimization (CPO), which explicitly specifies preference scores for different objectives, thereby guiding the model to generate responses that meet the requirements. Our experimental analysis reveals that the aligned models can provide responses that match various preferences among t
    
[^2]: GPSINDy: 数据驱动的动力学系统方程发现

    GPSINDy: Data-Driven Discovery of Equations of Motion. (arXiv:2309.11076v1 [cs.LG])

    [http://arxiv.org/abs/2309.11076](http://arxiv.org/abs/2309.11076)

    GPSINDy是一种数据驱动的方法，通过将高斯过程回归与SINDy相结合，能够从噪声数据中发现非线性动力学系统模型，并在实验证明了其在系统动态和预测未来轨迹方面的改进性能。

    

    本论文考虑从有噪声数据中发现动力学系统模型的问题。已知噪声存在对符号回归算法来说是一个重要问题。我们将高斯过程回归（一种非参数学习方法）与SINDy（一种参数学习方法）相结合，从数据中识别非线性动力学系统。我们的方法具有简单性和与SINDy相比在有噪声数据上表现出更好的鲁棒性的优点。我们在Lotka-Volterra模型和仿真中的单轮车动态模型上以及在使用硬件数据的NVIDIA JetRacer系统上展示了我们的方法。我们展示了相较于SINDy，我们的方法在发现系统动态和预测未来轨迹方面的改进性能。

    In this paper, we consider the problem of discovering dynamical system models from noisy data. The presence of noise is known to be a significant problem for symbolic regression algorithms. We combine Gaussian process regression, a nonparametric learning method, with SINDy, a parametric learning approach, to identify nonlinear dynamical systems from data. The key advantages of our proposed approach are its simplicity coupled with the fact that it demonstrates improved robustness properties with noisy data over SINDy. We demonstrate our proposed approach on a Lotka-Volterra model and a unicycle dynamic model in simulation and on an NVIDIA JetRacer system using hardware data. We demonstrate improved performance over SINDy for discovering the system dynamics and predicting future trajectories.
    

