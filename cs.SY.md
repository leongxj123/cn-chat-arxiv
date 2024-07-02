# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A General Verification Framework for Dynamical and Control Models via Certificate Synthesis.](http://arxiv.org/abs/2309.06090) | 这个论文提出了一个通用的框架来通过证书合成验证动态和控制模型。研究者们提供了一种自动化方法来设计控制器并分析复杂规范。这个方法利用神经网络和SMT求解器来提供候选控制和证书函数，并为控制的安全学习领域做出了贡献。 |
| [^2] | [A Convex Hull Cheapest Insertion Heuristic for the Non-Euclidean TSP.](http://arxiv.org/abs/2302.06582) | 本文提出了一种适用于非欧几里德旅行商问题的凸包最便宜插入启发式解法，通过使用多维缩放将非欧几里德空间的点近似到欧几里德空间，生成了初始化算法的凸包。在评估中发现，该算法在大多数情况下优于最邻近算法。 |
| [^3] | [Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment.](http://arxiv.org/abs/2208.13065) | 本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。 |

# 详细

[^1]: 通过证书合成的动态与控制模型的通用验证框架

    A General Verification Framework for Dynamical and Control Models via Certificate Synthesis. (arXiv:2309.06090v1 [eess.SY])

    [http://arxiv.org/abs/2309.06090](http://arxiv.org/abs/2309.06090)

    这个论文提出了一个通用的框架来通过证书合成验证动态和控制模型。研究者们提供了一种自动化方法来设计控制器并分析复杂规范。这个方法利用神经网络和SMT求解器来提供候选控制和证书函数，并为控制的安全学习领域做出了贡献。

    

    控制论的一个新兴分支专门研究证书学习，涉及对自主或控制模型的所需（可能是复杂的）系统行为的规范，并通过基于函数的证明进行分析验证。然而，满足这些复杂要求的控制器的合成通常是一个非常困难的任务，可能超出了大多数专家控制工程师的能力。因此，需要自动技术能够设计控制器并分析各种复杂规范。在本文中，我们提供了一个通用框架来编码系统规范并定义相应的证书，并提出了一种自动化方法来正式合成控制器和证书。我们的方法为控制的安全学习领域做出了贡献，利用神经网络的灵活性提供候选的控制和证书函数，同时使用SMT求解器来提供形式化的保证。

    An emerging branch of control theory specialises in certificate learning, concerning the specification of a desired (possibly complex) system behaviour for an autonomous or control model, which is then analytically verified by means of a function-based proof. However, the synthesis of controllers abiding by these complex requirements is in general a non-trivial task and may elude the most expert control engineers. This results in a need for automatic techniques that are able to design controllers and to analyse a wide range of elaborate specifications. In this paper, we provide a general framework to encode system specifications and define corresponding certificates, and we present an automated approach to formally synthesise controllers and certificates. Our approach contributes to the broad field of safe learning for control, exploiting the flexibility of neural networks to provide candidate control and certificate functions, whilst using SMT-solvers to offer a formal guarantee of co
    
[^2]: 非欧几里德旅行商问题的凸包最便宜插入启发式解法

    A Convex Hull Cheapest Insertion Heuristic for the Non-Euclidean TSP. (arXiv:2302.06582v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.06582](http://arxiv.org/abs/2302.06582)

    本文提出了一种适用于非欧几里德旅行商问题的凸包最便宜插入启发式解法，通过使用多维缩放将非欧几里德空间的点近似到欧几里德空间，生成了初始化算法的凸包。在评估中发现，该算法在大多数情况下优于最邻近算法。

    

    众所周知，凸包最便宜插入启发式算法可以在欧几里德空间中产生良好的旅行商问题解决方案，但还未在非欧几里德情况下进行扩展。为了解决非欧几里德空间中处理障碍物的困难，提出的改进方法使用多维缩放将这些点首先近似到欧几里德空间，从而可以生成初始化算法的凸包。通过修改TSPLIB基准数据集，向其中添加不可通过的分割器来产生非欧几里德空间，评估了所提出的算法。在所研究的案例中，该算法表现出优于常用的最邻近算法的性能，达到96%的情况。

    The convex hull cheapest insertion heuristic is known to generate good solutions to the Traveling Salesperson Problem in Euclidean spaces, but it has not been extended to the non-Euclidean case. To address the difficulty of dealing with obstacles in the non-Euclidean space, the proposed adaptation uses multidimensional scaling to first approximate these points in a Euclidean space, thereby enabling the generation of the convex hull that initializes the algorithm. To evaluate the proposed algorithm, the TSPLIB benchmark data-set is modified by adding impassable separators that produce non-Euclidean spaces. The algorithm is demonstrated to outperform the commonly used Nearest Neighbor algorithm in 96% of the cases studied.
    
[^3]: 改善运营经济学：基于双层 MIP 的闭环预测优化框架来预测机组组合的操作计划

    Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment. (arXiv:2208.13065v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2208.13065](http://arxiv.org/abs/2208.13065)

    本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。

    

    通常，系统操作员在开环预测优化过程中进行电力系统的经济运行：首先预测可再生能源(RES)的可用性和系统储备需求；根据这些预测，系统操作员解决诸如机组组合(UC)的优化模型，以确定相应的经济运行计划。然而，这种开环过程可能会实质性地损害操作经济性，因为它的预测器目光短浅地寻求改善即时的统计预测误差，而不是最终的操作成本。为此，本文提出了一个闭环预测优化框架，提供一种预测机组组合以改善操作经济性的方法。首先，利用双层混合整数规划模型针对最佳系统操作训练成本导向的预测器。上层基于其引起的操作成本来训练 RES 和储备预测器；下层则在给定预测的 RES 和储备的情况下，依据最佳操作原则求解 UC。这两个层级通过反馈环路进行交互性互动，直到收敛为止。在修改后的IEEE 24-bus系统上的数值实验表明，与三种最先进的 UC 基准线相比，所提出的框架具有高效性和有效性。

    Generally, system operators conduct the economic operation of power systems in an open-loop predict-then-optimize process: the renewable energy source (RES) availability and system reserve requirements are first predicted; given the predictions, system operators solve optimization models such as unit commitment (UC) to determine the economical operation plans accordingly. However, such an open-loop process could essentially compromise the operation economics because its predictors myopically seek to improve the immediate statistical prediction errors instead of the ultimate operation cost. To this end, this paper presents a closed-loop predict-and-optimize framework, offering a prescriptive UC to improve the operation economics. First, a bilevel mixed-integer programming model is leveraged to train cost-oriented predictors tailored for optimal system operations: the upper level trains the RES and reserve predictors based on their induced operation cost; the lower level, with given pred
    

