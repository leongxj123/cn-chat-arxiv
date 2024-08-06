# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deterministic Impartial Selection with Weights.](http://arxiv.org/abs/2310.14991) | 该研究提供了带权力的确定性公正机制，通过引入权重，改进了以前无权重设置下的逼近比率，并展示了该机制可以适用于公正分配问题。 |

# 详细

[^1]: 带权力的确定性公正选择

    Deterministic Impartial Selection with Weights. (arXiv:2310.14991v1 [cs.GT])

    [http://arxiv.org/abs/2310.14991](http://arxiv.org/abs/2310.14991)

    该研究提供了带权力的确定性公正机制，通过引入权重，改进了以前无权重设置下的逼近比率，并展示了该机制可以适用于公正分配问题。

    

    在公正选择问题中，基于代理人投票选取一个固定大小为$k$的代理人子集。如果没有代理人可以通过改变自己的投票来影响自己被选择的机会，则选择机制是公正的。如果对于每个实例，所选子集所接收的票数与获得最高票数的大小为$k$的子集所接收的票数的比率至少为$\alpha$的一部分，则该机制是$\alpha$-最优的。我们在一个更一般的设置中研究了带权力的确定性公正机制，并提供了首个逼近保证，大约为$1/\lceil 2n/k\rceil$。当要选择的代理人数量相对于总代理人数量足够大时，这比以前已知的无权重设置的逼近比率$1/k$有所改进。我们进一步证明了我们的机制可以适应公正分配问题，即在其中有多个集合需要选出代理人。

    In the impartial selection problem, a subset of agents up to a fixed size $k$ among a group of $n$ is to be chosen based on votes cast by the agents themselves. A selection mechanism is impartial if no agent can influence its own chance of being selected by changing its vote. It is $\alpha$-optimal if, for every instance, the ratio between the votes received by the selected subset is at least a fraction of $\alpha$ of the votes received by the subset of size $k$ with the highest number of votes. We study deterministic impartial mechanisms in a more general setting with arbitrarily weighted votes and provide the first approximation guarantee, roughly $1/\lceil 2n/k\rceil$. When the number of agents to select is large enough compared to the total number of agents, this yields an improvement on the previously best known approximation ratio of $1/k$ for the unweighted setting. We further show that our mechanism can be adapted to the impartial assignment problem, in which multiple sets of u
    

