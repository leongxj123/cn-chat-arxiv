# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic contextual bandits with graph feedback: from independence number to MAS number](https://arxiv.org/abs/2402.18591) | 本文研究了具有图反馈的上下文赌博问题，提出了一个刻画学习极限的图论量 $\beta_M(G)$，并建立了对应的遗憾下限。 |
| [^2] | [Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance.](http://arxiv.org/abs/2310.03722) | 本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。 |
| [^3] | [The ODE Method for Asymptotic Statistics in Stochastic Approximation and Reinforcement Learning.](http://arxiv.org/abs/2110.14427) | 本文提出了一种称为ODE方法的渐近统计方法解决$d$维随机逼近递归的问题，证明了其收敛性和中心极限定理，为强化学习等领域的应用提供了有力的理论支持。 |

# 详细

[^1]: 具有图反馈的随机上下文赌博：从独立数到MAS数

    Stochastic contextual bandits with graph feedback: from independence number to MAS number

    [https://arxiv.org/abs/2402.18591](https://arxiv.org/abs/2402.18591)

    本文研究了具有图反馈的上下文赌博问题，提出了一个刻画学习极限的图论量 $\beta_M(G)$，并建立了对应的遗憾下限。

    

    我们考虑具有图反馈的上下文赌博，在这类互动学习问题中，具有比普通上下文赌博更丰富结构，其中采取一个行动将在所有情境下揭示所有相邻行动的奖励。与多臂赌博设置不同，多文献已经对图反馈的理解进行了全面探讨，但在上下文赌博对应部分仍有许多未被探讨的地方。在本文中，我们通过建立一个遗憾下限 $\Omega(\sqrt{\beta_M(G) T})$ 探究了这个问题，其中 $M$ 是情境数，$G$ 是反馈图，$\beta_M(G)$ 是我们提出的表征该问题类的基础学习限制的图论量。有趣的是，$\beta_M(G)$ 在 $\alpha(G)$ (图的独立数) 和 $\mathsf{m}(G)$ (图的最大无环子图（MAS）数) 之间插值。

    arXiv:2402.18591v1 Announce Type: new  Abstract: We consider contextual bandits with graph feedback, a class of interactive learning problems with richer structures than vanilla contextual bandits, where taking an action reveals the rewards for all neighboring actions in the feedback graph under all contexts. Unlike the multi-armed bandits setting where a growing literature has painted a near-complete understanding of graph feedback, much remains unexplored in the contextual bandits counterpart. In this paper, we make inroads into this inquiry by establishing a regret lower bound $\Omega(\sqrt{\beta_M(G) T})$, where $M$ is the number of contexts, $G$ is the feedback graph, and $\beta_M(G)$ is our proposed graph-theoretical quantity that characterizes the fundamental learning limit for this class of problems. Interestingly, $\beta_M(G)$ interpolates between $\alpha(G)$ (the independence number of the graph) and $\mathsf{m}(G)$ (the maximum acyclic subgraph (MAS) number of the graph) as 
    
[^2]: 未知方差下的高斯均值的任意有效T检验和置信序列

    Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance. (arXiv:2310.03722v1 [math.ST])

    [http://arxiv.org/abs/2310.03722](http://arxiv.org/abs/2310.03722)

    本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。

    

    在1976年，Lai构造了一个非平凡的均值$\mu$的高斯分布的置信序列，该分布的方差$\sigma$是未知的。他使用了关于$\sigma$的不适当（右Haar）混合和关于$\mu$的不适当（平坦）混合。在本文中，我们详细说明了他构建的细节，其中使用了广义的不可积分鞅和扩展的维尔不等式。尽管这确实产生了一个顺序T检验，但由于他的鞅不可积分，它并没有产生一个“e-process”。在本文中，我们为相同的设置开发了两个新的“e-process”和置信序列：一个是在缩减滤波器中的测试鞅，另一个是在规范数据滤波器中的“e-process”。这些分别是通过将Lai的平坦混合替换为高斯混合，并将对$\sigma$的右Haar混合替换为在零空间下的最大似然估计，就像在通用推断中一样。我们还分析了所得结果的宽度。

    In 1976, Lai constructed a nontrivial confidence sequence for the mean $\mu$ of a Gaussian distribution with unknown variance $\sigma$. Curiously, he employed both an improper (right Haar) mixture over $\sigma$ and an improper (flat) mixture over $\mu$. Here, we elaborate carefully on the details of his construction, which use generalized nonintegrable martingales and an extended Ville's inequality. While this does yield a sequential t-test, it does not yield an ``e-process'' (due to the nonintegrability of his martingale). In this paper, we develop two new e-processes and confidence sequences for the same setting: one is a test martingale in a reduced filtration, while the other is an e-process in the canonical data filtration. These are respectively obtained by swapping Lai's flat mixture for a Gaussian mixture, and swapping the right Haar mixture over $\sigma$ with the maximum likelihood estimate under the null, as done in universal inference. We also analyze the width of resulting 
    
[^3]: 随机逼近和强化学习中渐近统计的ODE方法

    The ODE Method for Asymptotic Statistics in Stochastic Approximation and Reinforcement Learning. (arXiv:2110.14427v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2110.14427](http://arxiv.org/abs/2110.14427)

    本文提出了一种称为ODE方法的渐近统计方法解决$d$维随机逼近递归的问题，证明了其收敛性和中心极限定理，为强化学习等领域的应用提供了有力的理论支持。

    

    本文研究了$d$维随机逼近递归$$\theta_{n+1}=\theta_n+\alpha_{n+1}f(\theta_n, \Phi_{n+1})$$其中$\Phi$是一个在一般状态空间$\textsf{X}$上具有平稳分布$\pi$的几何遍历马尔可夫链，$f：\Re^d\times\textsf{X}\to\Re^d$。在称为（DV3）的Donsker-Varadhan Lyapunov漂移条件的一种版本和对具有向量场$\bar{f}(\theta)=\textsf{E}[f(\theta,\Phi)]$以及$\Phi\sim\pi$的均值流的稳定性条件下，建立了主要结果。(i) $\{\theta_n\}$以概率1和$L_4$收敛于$\bar{f}(\theta)$的唯一根$\theta^*$。(ii) 建立了泛函中心极限定理，以及归一化误差一维中心极限定理。(iii) 对于归一化版本$z_n{=:} \sqrt{n} (\theta^{\text{PR}}_n -\theta^*)$的平均参数$\theta^{\text{PR}}_n {=:} n^{-1} \sum_{k=1}^n\theta_k$ ，在步长的标准假设下，建立了中心极限定理。

    The paper concerns the $d$-dimensional stochastic approximation recursion, $$ \theta_{n+1}= \theta_n + \alpha_{n + 1} f(\theta_n, \Phi_{n+1}) $$ in which $\Phi$ is a geometrically ergodic Markov chain on a general state space $\textsf{X}$ with stationary distribution $\pi$, and $f:\Re^d\times\textsf{X}\to\Re^d$.  The main results are established under a version of the Donsker-Varadhan Lyapunov drift condition known as (DV3), and a stability condition for the mean flow with vector field $\bar{f}(\theta)=\textsf{E}[f(\theta,\Phi)]$, with $\Phi\sim\pi$.  (i) $\{ \theta_n\}$ is convergent a.s. and in $L_4$ to the unique root $\theta^*$ of $\bar{f}(\theta)$.  (ii) A functional CLT is established, as well as the usual one-dimensional CLT for the normalized error.  (iii) The CLT holds for the normalized version, $z_n{=:} \sqrt{n} (\theta^{\text{PR}}_n -\theta^*)$, of the averaged parameters, $\theta^{\text{PR}}_n {=:} n^{-1} \sum_{k=1}^n\theta_k$, subject to standard assumptions on the step-s
    

