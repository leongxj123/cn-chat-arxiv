# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Model-Free LQR Control over Rate-Limited Channels.](http://arxiv.org/abs/2401.01258) | 这篇论文研究了在速率限制通道上实现模型无关的LQR控制的问题。通过引入自适应量化梯度下降（AQGD）算法，作者证明了在噪声电路中可以实现控制问题的解决。 |
| [^2] | [Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation.](http://arxiv.org/abs/2210.05918) | 本研究通过引入尾平均和正则化技术，对时序差异(TD)学习算法进行了有限时间行为的研究。我们得出结论，尾平均TD能以最优速率 $O(1/t)$ 收敛，并且初始误差衰减速率更快。此外，正则化的TD版本在具有病态特征的问题上很有用。 |

# 详细

[^1]: 实现模型无关的通过速率限制通道的LQR控制

    Towards Model-Free LQR Control over Rate-Limited Channels. (arXiv:2401.01258v1 [math.OC])

    [http://arxiv.org/abs/2401.01258](http://arxiv.org/abs/2401.01258)

    这篇论文研究了在速率限制通道上实现模型无关的LQR控制的问题。通过引入自适应量化梯度下降（AQGD）算法，作者证明了在噪声电路中可以实现控制问题的解决。

    

    鉴于模型无关方法在许多问题设置中的控制设计方面取得的成功，自然而然地会问，如果利用实际的通信通道来传输梯度或策略，情况会如何改变。尽管由此产生的问题与网络控制系统中研究的公式有类似之处，但那个领域的丰富文献通常假定系统的模型是已知的。为了在模型无关控制设计和网络控制系统领域之间建立联系，我们提出了一个问题：\textit{是否可以通过速率限制的通道以模型无关的方式解决基本的控制问题-例如线性二次调节器（LQR）问题？}为了回答这个问题，我们研究了一个设置，其中一个工作代理通过一个无噪声信道以有限的位速率传输量化策略梯度（LQR成本）到一个服务器。我们提出了一种名为自适应量化梯度下降（AQGD）的新算法，并证明了

    Given the success of model-free methods for control design in many problem settings, it is natural to ask how things will change if realistic communication channels are utilized for the transmission of gradients or policies. While the resulting problem has analogies with the formulations studied under the rubric of networked control systems, the rich literature in that area has typically assumed that the model of the system is known. As a step towards bridging the fields of model-free control design and networked control systems, we ask: \textit{Is it possible to solve basic control problems - such as the linear quadratic regulator (LQR) problem - in a model-free manner over a rate-limited channel?} Toward answering this question, we study a setting where a worker agent transmits quantized policy gradients (of the LQR cost) to a server over a noiseless channel with a finite bit-rate. We propose a new algorithm titled Adaptively Quantized Gradient Descent (\texttt{AQGD}), and prove that
    
[^2]: 有限时间内使用线性函数逼近进行时序差异学习的分析：尾平均和正则化

    Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation. (arXiv:2210.05918v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.05918](http://arxiv.org/abs/2210.05918)

    本研究通过引入尾平均和正则化技术，对时序差异(TD)学习算法进行了有限时间行为的研究。我们得出结论，尾平均TD能以最优速率 $O(1/t)$ 收敛，并且初始误差衰减速率更快。此外，正则化的TD版本在具有病态特征的问题上很有用。

    

    本文研究了将流行的时序差异(TD)学习算法与尾平均相结合时的有限时间行为。我们在不需要关于底层投影TD不动点矩阵的特征值信息的步长选择下，推导了尾平均TD迭代的参数误差的有限时间界。我们的分析表明，尾平均TD以期望速率和高概率收敛于最优的 $O(1/t)$ 速率。此外，我们的界限展示了初始误差(偏差)的更快衰减速率，这是对所有迭代的平均值的改进。我们还提出并分析了一种结合正则化的TD变体。通过分析，我们得出结论认为正则化的TD版本在具有病态特征的问题上是有用的。

    We study the finite-time behaviour of the popular temporal difference (TD) learning algorithm when combined with tail-averaging. We derive finite time bounds on the parameter error of the tail-averaged TD iterate under a step-size choice that does not require information about the eigenvalues of the matrix underlying the projected TD fixed point. Our analysis shows that tail-averaged TD converges at the optimal $O\left(1/t\right)$ rate, both in expectation and with high probability. In addition, our bounds exhibit a sharper rate of decay for the initial error (bias), which is an improvement over averaging all iterates. We also propose and analyse a variant of TD that incorporates regularisation. From analysis, we conclude that the regularised version of TD is useful for problems with ill-conditioned features.
    

