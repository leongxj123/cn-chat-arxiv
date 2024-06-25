# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Slingshot Approach to Learning in Monotone Games.](http://arxiv.org/abs/2305.16610) | 本文提出了一种新的框架, 通过正则化游戏的支付或效用和更新投石索策略，无论是否存在噪声都能够实现在单调博弈中计算均衡。 |
| [^2] | [Game Transformations That Preserve Nash Equilibria or Best Response Sets.](http://arxiv.org/abs/2111.00076) | 本研究探讨了对N人博弈应用的游戏变换中，哪些变换可以保持最佳反应集或纳什均衡集。我们证明了正仿射变换可以保持最佳反应集。这个研究提供了一个明确的描述，说明哪些游戏变换可以保持最佳反应集或纳什均衡集。 |

# 详细

[^1]: 学习单调博弈的投石索方法

    A Slingshot Approach to Learning in Monotone Games. (arXiv:2305.16610v1 [cs.GT])

    [http://arxiv.org/abs/2305.16610](http://arxiv.org/abs/2305.16610)

    本文提出了一种新的框架, 通过正则化游戏的支付或效用和更新投石索策略，无论是否存在噪声都能够实现在单调博弈中计算均衡。

    

    本文解决了在单调博弈中计算均衡的问题。传统的遵循正则化领导者算法即使在双人零和游戏中也无法收敛到均衡。虽然已经提出了这些算法的乐观版本并具有最后迭代的收敛保证，但它们需要无噪声的梯度反馈。为了克服这个限制，我们提出了一个新的框架，即使在存在噪声的情况下也能实现最后一次迭代的收敛。我们的关键思想是扰动或正则化游戏的支付或效用。这种扰动有助于将当前策略拉向一个锚定策略，我们称之为“投石索”策略。首先，我们建立了框架的收敛速度，从而获得靠近均衡点的稳定点，无论是否存在噪声。接下来，我们介绍了一种方法，定期更新投石索策略和当前策略。我们将这种方法解释为近端p

    In this paper, we address the problem of computing equilibria in monotone games. The traditional Follow the Regularized Leader algorithms fail to converge to an equilibrium even in two-player zero-sum games. Although optimistic versions of these algorithms have been proposed with last-iterate convergence guarantees, they require noiseless gradient feedback. To overcome this limitation, we present a novel framework that achieves last-iterate convergence even in the presence of noise. Our key idea involves perturbing or regularizing the payoffs or utilities of the games. This perturbation serves to pull the current strategy to an anchored strategy, which we refer to as a {\it slingshot} strategy. First, we establish the convergence rates of our framework to a stationary point near an equilibrium, regardless of the presence or absence of noise. Next, we introduce an approach to periodically update the slingshot strategy with the current strategy. We interpret this approach as a proximal p
    
[^2]: 保持纳什均衡或最佳反应集的游戏变换

    Game Transformations That Preserve Nash Equilibria or Best Response Sets. (arXiv:2111.00076v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2111.00076](http://arxiv.org/abs/2111.00076)

    本研究探讨了对N人博弈应用的游戏变换中，哪些变换可以保持最佳反应集或纳什均衡集。我们证明了正仿射变换可以保持最佳反应集。这个研究提供了一个明确的描述，说明哪些游戏变换可以保持最佳反应集或纳什均衡集。

    

    在同时非合作博弈的文献中，广泛使用的事实是，效用收益的正仿射（线性）变换既不改变最佳反应集，也不改变纳什均衡集。我们研究了哪些其他游戏变换在应用于任意N人游戏（N≥2）时也具有这两种属性之一：（i）纳什均衡集保持不变；（ii）最佳反应集保持不变。对于以玩家和策略为基础的游戏变换，我们证明（i）意味着（ii），具有属性（ii）的变换必须是正仿射的。得到的等价链明确描述了那些总是保持纳什均衡集（或最佳反应集）的游戏变换。同时，我们获得了正仿射变换类的两个新特征描述。

    In the literature on simultaneous non-cooperative games, it is a widely used fact that a positive affine (linear) transformation of the utility payoffs neither changes the best response sets nor the Nash equilibrium set. We investigate which other game transformations also possess one of these two properties when being applied to an arbitrary N-player game (N >= 2):  (i) The Nash equilibrium set stays the same.  (ii) The best response sets stay the same.  For game transformations that operate player-wise and strategy-wise, we prove that (i) implies (ii) and that transformations with property (ii) must be positive affine. The resulting equivalence chain gives an explicit description of all those game transformations that always preserve the Nash equilibrium set (or, respectively, the best response sets). Simultaneously, we obtain two new characterizations of the class of positive affine transformations.
    

