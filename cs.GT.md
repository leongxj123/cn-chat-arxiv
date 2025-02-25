# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Manipulate under Limited Information.](http://arxiv.org/abs/2401.16412) | 本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。 |
| [^2] | [From External to Swap Regret 2.0: An Efficient Reduction and Oblivious Adversary for Large Action Spaces.](http://arxiv.org/abs/2310.19786) | 通过新的约化方法，我们改进了经典的Swap遗憾最小化算法，并提供了一个无外部遗憾算法的替代方法。对于学习专家建议问题，我们的算法可以在较少的轮数和更低的复杂度下达到相同的Swap遗憾限制。 |

# 详细

[^1]: 学习在有限信息下进行操纵

    Learning to Manipulate under Limited Information. (arXiv:2401.16412v1 [cs.AI])

    [http://arxiv.org/abs/2401.16412](http://arxiv.org/abs/2401.16412)

    本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。

    

    根据社会选择理论的经典结果，任何合理的偏好投票方法有时会给个体提供报告不真实偏好的激励。对于比较投票方法来说，不同投票方法在多大程度上更或者更少抵抗这种策略性操纵已成为一个关键考虑因素。在这里，我们通过神经网络在不同规模下对限制信息下学习如何利用给定投票方法进行操纵的成功程度来衡量操纵的抵抗力。我们训练了将近40,000个不同规模的神经网络来对抗8种不同的投票方法，在6种限制信息情况下，进行包含5-21名选民和3-6名候选人的委员会规模选举的操纵。我们发现，一些投票方法，如Borda方法，在有限信息下可以被神经网络高度操纵，而其他方法，如Instant Runoff方法，虽然被一个理想的操纵者利润化操纵，但在有限信息下不会受到操纵。

    By classic results in social choice theory, any reasonable preferential voting method sometimes gives individuals an incentive to report an insincere preference. The extent to which different voting methods are more or less resistant to such strategic manipulation has become a key consideration for comparing voting methods. Here we measure resistance to manipulation by whether neural networks of varying sizes can learn to profitably manipulate a given voting method in expectation, given different types of limited information about how other voters will vote. We trained nearly 40,000 neural networks of 26 sizes to manipulate against 8 different voting methods, under 6 types of limited information, in committee-sized elections with 5-21 voters and 3-6 candidates. We find that some voting methods, such as Borda, are highly manipulable by networks with limited information, while others, such as Instant Runoff, are not, despite being quite profitably manipulated by an ideal manipulator with
    
[^2]: 从外部到Swap遗憾2.0：针对大动作空间的高效约化和无知对手

    From External to Swap Regret 2.0: An Efficient Reduction and Oblivious Adversary for Large Action Spaces. (arXiv:2310.19786v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.19786](http://arxiv.org/abs/2310.19786)

    通过新的约化方法，我们改进了经典的Swap遗憾最小化算法，并提供了一个无外部遗憾算法的替代方法。对于学习专家建议问题，我们的算法可以在较少的轮数和更低的复杂度下达到相同的Swap遗憾限制。

    

    我们提供了一种新颖的从Swap遗憾最小化到外部遗憾最小化的约化方法，改进了Blum-Mansour和Stolz-Lugosi的经典约化方法，不需要行为空间的有限性。我们证明，只要存在某个假设类的无外部遗憾算法，就必然存在相同类别的无Swap遗憾算法。对于学习专家建议问题，我们的结果意味着可以保证在$\log(N)^{O(1/\epsilon)}$轮后，每次迭代复杂度为$O(N)$的情况下，Swap遗憾被限定为$\epsilon$，而Blum-Mansour和Stolz-Lugosi的经典约化方法需要$O(N/\epsilon^2)$轮和至少$\Omega(N^2)$的每次迭代复杂度。我们的结果伴随着一个相关的下界，与[BM07]不同，这个下界适用于无知和$\ell_1$-受限的对手和可以利用这个下界的学习者。

    We provide a novel reduction from swap-regret minimization to external-regret minimization, which improves upon the classical reductions of Blum-Mansour [BM07] and Stolz-Lugosi [SL05] in that it does not require finiteness of the space of actions. We show that, whenever there exists a no-external-regret algorithm for some hypothesis class, there must also exist a no-swap-regret algorithm for that same class. For the problem of learning with expert advice, our result implies that it is possible to guarantee that the swap regret is bounded by {\epsilon} after $\log(N)^{O(1/\epsilon)}$ rounds and with $O(N)$ per iteration complexity, where $N$ is the number of experts, while the classical reductions of Blum-Mansour and Stolz-Lugosi require $O(N/\epsilon^2)$ rounds and at least $\Omega(N^2)$ per iteration complexity. Our result comes with an associated lower bound, which -- in contrast to that in [BM07] -- holds for oblivious and $\ell_1$-constrained adversaries and learners that can emplo
    

