# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling](https://arxiv.org/abs/2402.05098) | 本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。 |
| [^2] | [Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network.](http://arxiv.org/abs/2305.19366) | 本文提出了在单一生成流网络中联合建模贝叶斯网络结构和参数的方法，包括非离散样本空间，提高了贝叶斯网络局部概率模型的灵活性。 |
| [^3] | [Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets.](http://arxiv.org/abs/2305.17010) | 本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。 |
| [^4] | [Trajectory balance: Improved credit assignment in GFlowNets.](http://arxiv.org/abs/2201.13259) | GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。 |

# 详细

[^1]: 关于分散推断模型的扩散模型：基准测试和改进随机控制和采样

    On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling

    [https://arxiv.org/abs/2402.05098](https://arxiv.org/abs/2402.05098)

    本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。

    

    我们研究了训练扩散模型以从给定的非标准化密度或能量函数分布中采样的问题。我们对几种扩散结构推断方法进行了基准测试，包括基于模拟的变分方法和离策略方法（连续生成流网络）。我们的结果揭示了现有算法的相对优势，同时对过去的研究提出了一些质疑。我们还提出了一种新颖的离策略方法探索策略，基于目标空间中的局部搜索和回放缓冲区的使用，并证明它可以改善各种目标分布上的样本质量。我们研究的采样方法和基准测试的代码已公开在https://github.com/GFNOrg/gfn-diffusion，作为未来在分散推断模型上工作的基础。

    We study the problem of training diffusion models to sample from a distribution with a given unnormalized density or energy function. We benchmark several diffusion-structured inference methods, including simulation-based variational approaches and off-policy methods (continuous generative flow networks). Our results shed light on the relative advantages of existing algorithms while bringing into question some claims from past work. We also propose a novel exploration strategy for off-policy methods, based on local search in the target space with the use of a replay buffer, and show that it improves the quality of samples on a variety of target distributions. Our code for the sampling methods and benchmarks studied is made public at https://github.com/GFNOrg/gfn-diffusion as a base for future work on diffusion models for amortized inference.
    
[^2]: 单一生成流网络中的图结构与参数的联合贝叶斯推理

    Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network. (arXiv:2305.19366v1 [cs.LG])

    [http://arxiv.org/abs/2305.19366](http://arxiv.org/abs/2305.19366)

    本文提出了在单一生成流网络中联合建模贝叶斯网络结构和参数的方法，包括非离散样本空间，提高了贝叶斯网络局部概率模型的灵活性。

    

    生成流网络是一类对离散和结构化样本空间进行建模的生成模型。先前的研究已将其应用于推断给定观测数据的贝叶斯网络的有向无环图（DAG）的边缘后验分布。本文基于最近的研究进展，在非离散样本空间上将此框架扩展到联合后验分布的建模，不仅包括贝叶斯网络的结构，还考虑了其条件概率分布的参数。

    Generative Flow Networks (GFlowNets), a class of generative models over discrete and structured sample spaces, have been previously applied to the problem of inferring the marginal posterior distribution over the directed acyclic graph (DAG) of a Bayesian Network, given a dataset of observations. Based on recent advances extending this framework to non-discrete sample spaces, we propose in this paper to approximate the joint posterior over not only the structure of a Bayesian Network, but also the parameters of its conditional probability distributions. We use a single GFlowNet whose sampling policy follows a two-phase process: the DAG is first generated sequentially one edge at a time, and then the corresponding parameters are picked once the full structure is known. Since the parameters are included in the posterior distribution, this leaves more flexibility for the local probability models of the Bayesian Network, making our approach applicable even to non-linear models parametrized
    
[^3]: 利用GFlowNets解决图形组合优化问题

    Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets. (arXiv:2305.17010v1 [cs.LG])

    [http://arxiv.org/abs/2305.17010](http://arxiv.org/abs/2305.17010)

    本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。

    

    组合优化问题通常是NP难题，因此不适用于精确算法，这使它们成为应用机器学习方法的理想领域。这些问题中高度结构化的限制可能会直接阻碍优化或采样解决方案的空间。另一方面，GFlowNets最近被发现是一种强大的机器，可以顺序地从复合非规范化密度中有效地采样，并具有在CO中分摊此类解决方案搜索过程以及生成不同的解决方案候选项的潜力。在本文中，我们设计了适用于不同组合问题的马尔科夫决策过程（MDP），并提出训练有条件的GFlowNets从解空间中采样的策略。还开发了高效的训练技术来受益于远程信用分配。通过对各种使用合成和实际数据的不同CO任务的广泛实验，我们证明了GFlowNet策略可以有效地找到高质量的解。

    Combinatorial optimization (CO) problems are often NP-hard and thus out of reach for exact algorithms, making them a tempting domain to apply machine learning methods. The highly structured constraints in these problems can hinder either optimization or sampling directly in the solution space. On the other hand, GFlowNets have recently emerged as a powerful machinery to efficiently sample from composite unnormalized densities sequentially and have the potential to amortize such solution-searching processes in CO, as well as generate diverse solution candidates. In this paper, we design Markov decision processes (MDPs) for different combinatorial problems and propose to train conditional GFlowNets to sample from the solution space. Efficient training techniques are also developed to benefit long-range credit assignment. Through extensive experiments on a variety of different CO tasks with synthetic and realistic data, we demonstrate that GFlowNet policies can efficiently find high-quali
    
[^4]: 轨迹平衡：改进了GFlowNets中的信用分配

    Trajectory balance: Improved credit assignment in GFlowNets. (arXiv:2201.13259v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.13259](http://arxiv.org/abs/2201.13259)

    GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。

    

    生成流网络（GFlowNets）是一种学习使用动作序列生成组合对象（如图形或字符串）的随机策略的方法，其中许多可能的动作序列可能导致相同的对象。我们发现先前提出的GFlowNets学习目标，即流匹配和详细平衡，类似于时间差分学习，容易在长的动作序列中出现信用传播效率低下的问题。因此，我们提出了一种新的学习目标，即轨迹平衡，作为先前使用目标的更高效的替代方法。我们证明了轨迹平衡目标的任何全局极小值可以定义一个从目标分布精确采样的策略。在四个不同领域的实验中，我们从实证上证明了轨迹平衡目标对于GFlowNet收敛性、生成样本的多样性以及对长动作序列和噪声的鲁棒性的益处。

    Generative flow networks (GFlowNets) are a method for learning a stochastic policy for generating compositional objects, such as graphs or strings, from a given unnormalized density by sequences of actions, where many possible action sequences may lead to the same object. We find previously proposed learning objectives for GFlowNets, flow matching and detailed balance, which are analogous to temporal difference learning, to be prone to inefficient credit propagation across long action sequences. We thus propose a new learning objective for GFlowNets, trajectory balance, as a more efficient alternative to previously used objectives. We prove that any global minimizer of the trajectory balance objective can define a policy that samples exactly from the target distribution. In experiments on four distinct domains, we empirically demonstrate the benefits of the trajectory balance objective for GFlowNet convergence, diversity of generated samples, and robustness to long action sequences and
    

