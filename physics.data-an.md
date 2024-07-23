# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resolution of Simpson's paradox via the common cause principle](https://arxiv.org/abs/2403.00957) | 通过对共同原因$C$进行条件设定，解决了辛普森悖论，推广了悖论，并表明在二元共同原因$C$上进行条件设定的关联方向与原始$B$上进行条件设定相同 |
| [^2] | [Bayesian identification of nonseparable Hamiltonians with multiplicative noise using deep learning and reduced-order modeling.](http://arxiv.org/abs/2401.12476) | 本文提出了一种用于学习非分离哈密顿系统的结构保持的贝叶斯方法，可以处理统计相关的加性和乘性噪声，并且通过将结构保持方法纳入框架中，提供了对高维系统的高效识别。 |

# 详细

[^1]: 利用共因原则解决辛普森悖论

    Resolution of Simpson's paradox via the common cause principle

    [https://arxiv.org/abs/2403.00957](https://arxiv.org/abs/2403.00957)

    通过对共同原因$C$进行条件设定，解决了辛普森悖论，推广了悖论，并表明在二元共同原因$C$上进行条件设定的关联方向与原始$B$上进行条件设定相同

    

    辛普森悖论是建立两个事件$a_1$和$a_2$之间的概率关联时的障碍，给定第三个（潜在的）随机变量$B$。我们关注的情景是随机变量$A$（汇总了$a_1$、$a_2$及其补集）和$B$有一个可能未被观察到的共同原因$C$。或者，我们可以假设$C$将$A$从$B$中筛选出去。对于这种情况，正确的$a_1$和$a_2$之间的关联应该通过对$C$进行条件设定来定义。这一设置将原始辛普森悖论推广了。现在它的两个相互矛盾的选项简单地指的是两个特定且不同的原因$C$。我们表明，如果$B$和$C$是二进制的，$A$是四进制的（对于有效的辛普森悖论来说是最小且最常见的情况），在任何二元共同原因$C$上进行条件设定将建立与在原始$B$上进行条件设定相同的$a_1$和$a_2$之间的关联方向。

    arXiv:2403.00957v1 Announce Type: cross  Abstract: Simpson's paradox is an obstacle to establishing a probabilistic association between two events $a_1$ and $a_2$, given the third (lurking) random variable $B$. We focus on scenarios when the random variables $A$ (which combines $a_1$, $a_2$, and their complements) and $B$ have a common cause $C$ that need not be observed. Alternatively, we can assume that $C$ screens out $A$ from $B$. For such cases, the correct association between $a_1$ and $a_2$ is to be defined via conditioning over $C$. This set-up generalizes the original Simpson's paradox. Now its two contradicting options simply refer to two particular and different causes $C$. We show that if $B$ and $C$ are binary and $A$ is quaternary (the minimal and the most widespread situation for valid Simpson's paradox), the conditioning over any binary common cause $C$ establishes the same direction of the association between $a_1$ and $a_2$ as the conditioning over $B$ in the original
    
[^2]: 用深度学习和降阶建模进行贝叶斯非分离哈密顿系统的识别和多项式噪声 (arXiv:2401.12476v1 [stat.ML])

    Bayesian identification of nonseparable Hamiltonians with multiplicative noise using deep learning and reduced-order modeling. (arXiv:2401.12476v1 [stat.ML])

    [http://arxiv.org/abs/2401.12476](http://arxiv.org/abs/2401.12476)

    本文提出了一种用于学习非分离哈密顿系统的结构保持的贝叶斯方法，可以处理统计相关的加性和乘性噪声，并且通过将结构保持方法纳入框架中，提供了对高维系统的高效识别。

    

    本文提出了一种结构保持的贝叶斯方法，用于学习使用随机动力模型的非分离哈密顿系统，该系统允许统计相关的，矢量值的加性和乘性测量噪声。该方法由三个主要方面组成。首先，我们推导了一个用于评估贝叶斯后验中的似然函数所需的统计相关的，矢量值的加性和乘性噪声模型的高斯滤波器。其次，我们开发了一种新算法，用于对高维系统进行高效的贝叶斯系统识别。第三，我们演示了如何将结构保持方法纳入所提议的框架中，使用非分离哈密顿系统作为一个举例的系统类别。我们将贝叶斯方法与一种最先进的机器学习方法在一个典型的非分离哈密顿模型和带有小型噪声训练数据集的混沌双摆模型上进行了比较，实验结果表明

    This paper presents a structure-preserving Bayesian approach for learning nonseparable Hamiltonian systems using stochastic dynamic models allowing for statistically-dependent, vector-valued additive and multiplicative measurement noise. The approach is comprised of three main facets. First, we derive a Gaussian filter for a statistically-dependent, vector-valued, additive and multiplicative noise model that is needed to evaluate the likelihood within the Bayesian posterior. Second, we develop a novel algorithm for cost-effective application of Bayesian system identification to high-dimensional systems. Third, we demonstrate how structure-preserving methods can be incorporated into the proposed framework, using nonseparable Hamiltonians as an illustrative system class. We compare the Bayesian method to a state-of-the-art machine learning method on a canonical nonseparable Hamiltonian model and a chaotic double pendulum model with small, noisy training datasets. The results show that us
    

