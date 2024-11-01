# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the tightness of information-theoretic bounds on generalization error of learning algorithms.](http://arxiv.org/abs/2303.14658) | 本文研究了学习算法泛化误差信息理论界限的紧密性。研究表明，通过适当的假设，可以在快速收敛速度下使用信息理论量$O(\lambda/n)$来上界估计泛化误差。 |

# 详细

[^1]: 关于学习算法泛化误差信息理论界限的紧密性研究

    On the tightness of information-theoretic bounds on generalization error of learning algorithms. (arXiv:2303.14658v1 [cs.IT])

    [http://arxiv.org/abs/2303.14658](http://arxiv.org/abs/2303.14658)

    本文研究了学习算法泛化误差信息理论界限的紧密性。研究表明，通过适当的假设，可以在快速收敛速度下使用信息理论量$O(\lambda/n)$来上界估计泛化误差。

    

    Russo和Xu提出了一种方法来证明学习算法的泛化误差可以通过信息度量进行上界估计。然而，这种收敛速度通常被认为是“慢”的，因为它的期望收敛速度的形式为$O(\sqrt{\lambda/n})$，其中$\lambda$是一些信息理论量。在本文中我们证明了根号并不一定意味着收敛速度慢，可以在适当的假设下使用这个界限来得到$O(\lambda/n)$的快速收敛速度。此外，我们确定了达到快速收敛速度的关键条件，即所谓的$(\eta,c)$-中心条件。在这个条件下，我们给出了学习算法泛化误差的信息理论界限。

    A recent line of works, initiated by Russo and Xu, has shown that the generalization error of a learning algorithm can be upper bounded by information measures. In most of the relevant works, the convergence rate of the expected generalization error is in the form of $O(\sqrt{\lambda/n})$ where $\lambda$ is some information-theoretic quantities such as the mutual information or conditional mutual information between the data and the learned hypothesis. However, such a learning rate is typically considered to be ``slow", compared to a ``fast rate" of $O(\lambda/n)$ in many learning scenarios. In this work, we first show that the square root does not necessarily imply a slow rate, and a fast rate result can still be obtained using this bound under appropriate assumptions. Furthermore, we identify the critical conditions needed for the fast rate generalization error, which we call the $(\eta,c)$-central condition. Under this condition, we give information-theoretic bounds on the generaliz
    

