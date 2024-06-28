# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control.](http://arxiv.org/abs/2311.07202) | 本研究提出了一种基于输入凸LSTM的基于Lyapunov的模型预测控制方法，通过减少收敛时间和缓解梯度消失/爆炸问题来改善MPC的性能。 |

# 详细

[^1]: 输入凸LSTM：一种快速基于Lyapunov模型预测控制的凸方法

    Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control. (arXiv:2311.07202v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.07202](http://arxiv.org/abs/2311.07202)

    本研究提出了一种基于输入凸LSTM的基于Lyapunov的模型预测控制方法，通过减少收敛时间和缓解梯度消失/爆炸问题来改善MPC的性能。

    

    利用输入凸神经网络（ICNN），基于ICNN的模型预测控制（MPC）通过在MPC框架中保持凸性成功实现全局最优解。然而，当前的ICNN架构存在梯度消失/爆炸问题，限制了它们作为复杂任务的深度神经网络的能力。此外，当前基于神经网络的MPC，包括传统的基于神经网络的MPC和基于ICNN的MPC，与基于第一原理模型的MPC相比面临较慢的收敛速度。在本研究中，我们利用ICNN的原理提出了一种新的基于输入凸LSTM的基于Lyapunov的MPC，旨在减少收敛时间、缓解梯度消失/爆炸问题并确保闭环稳定性。通过对非线性化学反应器的模拟研究，我们观察到了梯度消失/爆炸问题的缓解和收敛时间的减少，收敛时间平均降低了一定的百分之。

    Leveraging Input Convex Neural Networks (ICNNs), ICNN-based Model Predictive Control (MPC) successfully attains globally optimal solutions by upholding convexity within the MPC framework. However, current ICNN architectures encounter the issue of vanishing/exploding gradients, which limits their ability to serve as deep neural networks for complex tasks. Additionally, the current neural network-based MPC, including conventional neural network-based MPC and ICNN-based MPC, faces slower convergence speed when compared to MPC based on first-principles models. In this study, we leverage the principles of ICNNs to propose a novel Input Convex LSTM for Lyapunov-based MPC, with the specific goal of reducing convergence time and mitigating the vanishing/exploding gradient problem while ensuring closed-loop stability. From a simulation study of a nonlinear chemical reactor, we observed a mitigation of vanishing/exploding gradient problem and a reduction in convergence time, with a percentage de
    

