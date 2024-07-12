# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Playing Large Games with Oracles and AI Debate](https://arxiv.org/abs/2312.04792) | 本论文研究了在具有大量动作的重复游戏中实现遗憾最小化的问题，通过使用基于Oracle的算法，提出了一种高效的内部遗憾最小化算法，实现了在大型游戏中计算相关均衡的高效性。通过在AI安全辩论环境中的实验验证了算法的有效性。 |

# 详细

[^1]: 使用Oracle和AI辩论进行大型游戏的玩法

    Playing Large Games with Oracles and AI Debate

    [https://arxiv.org/abs/2312.04792](https://arxiv.org/abs/2312.04792)

    本论文研究了在具有大量动作的重复游戏中实现遗憾最小化的问题，通过使用基于Oracle的算法，提出了一种高效的内部遗憾最小化算法，实现了在大型游戏中计算相关均衡的高效性。通过在AI安全辩论环境中的实验验证了算法的有效性。

    

    我们考虑在具有大量动作的重复游戏中实现遗憾最小化。这种游戏在通过辩论确保AI安全的环境中是固有的，并且更一般地应用于动作基于语言的游戏中。现有的在线游戏算法需要多项式计算数量的动作，而对于大型游戏来说，这可能是难以实现的。因此，我们考虑使用基于Oracle的算法，因为Oracle自然地模拟了对AI代理的访问。通过对Oracle访问进行特征化，我们可以有效地实现内部和外部遗憾的最小化。我们提出了一种新颖的内部遗憾最小化算法，其遗憾和计算复杂度对数地依赖于动作数量。这意味着可以高效地基于Oracle计算大型游戏中的相关均衡。最后，我们通过在AI安全辩论环境中进行实验，展示了我们算法分析的好处。

    We consider regret minimization in repeated games with a very large number of actions. Such games are inherent in the setting of AI safety via debate, and more generally games whose actions are language-based. Existing algorithms for online game playing require computation polynomial in the number of actions, which can be prohibitive for large games.   We thus consider oracle-based algorithms, as oracles naturally model access to AI agents. With oracle access, we characterize when internal and external regret can be minimized efficiently. We give a novel efficient algorithm for internal regret minimization whose regret and computation complexity depend logarithmically on the number of actions. This implies efficient oracle-based computation of a correlated equilibrium in large games.   We conclude with experiments in the setting of AI Safety via Debate that shows the benefit of insights from our algorithmic analysis.
    

