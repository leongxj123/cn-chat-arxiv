# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Local Control Barrier Functions for Safety Control of Hybrid Systems.](http://arxiv.org/abs/2401.14907) | 该论文提出了一种学习启用的方法，能够构建本地控制屏障函数，以保证广泛类别的非线性混合动力系统的安全性。该方法是高效的，对参考控制器的干预最小，适用于大规模系统，并通过实证评估和比较案例展示了其功效和灵活性。 |
| [^2] | [Autonomous search of real-life environments combining dynamical system-based path planning and unsupervised learning.](http://arxiv.org/abs/2305.01834) | 本文提出了一种自动生成基于动态系统路径规划器和无监督机器学习技术相结合的算法，以克服混沌覆盖路径规划器的立即问题，并在模拟和实际环境中进行了测试，展示其在有限环境中实现自主搜索和覆盖的能力。 |

# 详细

[^1]: 学习本地控制屏障函数以实现混合系统的安全控制

    Learning Local Control Barrier Functions for Safety Control of Hybrid Systems. (arXiv:2401.14907v1 [cs.RO])

    [http://arxiv.org/abs/2401.14907](http://arxiv.org/abs/2401.14907)

    该论文提出了一种学习启用的方法，能够构建本地控制屏障函数，以保证广泛类别的非线性混合动力系统的安全性。该方法是高效的，对参考控制器的干预最小，适用于大规模系统，并通过实证评估和比较案例展示了其功效和灵活性。

    

    混合动力系统在实际的机器人应用中普遍存在，常涉及连续状态和离散状态切换。安全性是混合机器人系统的首要关注点。现有的混合系统的安全关键控制方法要么计算效率低下，对系统性能有损，要么仅适用于小规模系统。为了解决这些问题，在本文中，我们提出了一种学习启用的方法，用于构建本地控制屏障函数（CBFs），以保证广泛类别的非线性混合动力系统的安全性。最终，我们得到了一个安全的基于神经网络的CBF切换控制器。我们的方法在计算上高效，对任何参考控制器的干预最小，并适用于大规模系统。通过两个机器人示例（包括高维自主赛车案例），我们对我们的框架进行了实证评估，并与其他基于CBF的方法和模型预测控制进行了比较，展示了其功效和灵活性。

    Hybrid dynamical systems are ubiquitous as practical robotic applications often involve both continuous states and discrete switchings. Safety is a primary concern for hybrid robotic systems. Existing safety-critical control approaches for hybrid systems are either computationally inefficient, detrimental to system performance, or limited to small-scale systems. To amend these drawbacks, in this paper, we propose a learningenabled approach to construct local Control Barrier Functions (CBFs) to guarantee the safety of a wide class of nonlinear hybrid dynamical systems. The end result is a safe neural CBFbased switching controller. Our approach is computationally efficient, minimally invasive to any reference controller, and applicable to large-scale systems. We empirically evaluate our framework and demonstrate its efficacy and flexibility through two robotic examples including a high-dimensional autonomous racing case, against other CBF-based approaches and model predictive control.
    
[^2]: 基于动态系统路径规划和无监督学习的实时环境自主搜索

    Autonomous search of real-life environments combining dynamical system-based path planning and unsupervised learning. (arXiv:2305.01834v1 [cs.RO])

    [http://arxiv.org/abs/2305.01834](http://arxiv.org/abs/2305.01834)

    本文提出了一种自动生成基于动态系统路径规划器和无监督机器学习技术相结合的算法，以克服混沌覆盖路径规划器的立即问题，并在模拟和实际环境中进行了测试，展示其在有限环境中实现自主搜索和覆盖的能力。

    

    近年来取得了使用混沌覆盖路径规划器进行有限环境搜索和遍历的进展，但该领域的现状仍处于初级阶段，目前的实验工作尚未开发出可满足混沌覆盖路径规划器需要克服的立即问题的强大方法 。本文旨在提出一种自动生成基于动态系统路径规划器和无监督机器学习技术相结合的算法，以克服混沌覆盖路径规划器的立即问题，并在模拟和实际环境中进行测试，展示其在有限环境中实现自主搜索和覆盖的能力。

    In recent years, advancements have been made towards the goal of using chaotic coverage path planners for autonomous search and traversal of spaces with limited environmental cues. However, the state of this field is still in its infancy as there has been little experimental work done. Current experimental work has not developed robust methods to satisfactorily address the immediate set of problems a chaotic coverage path planner needs to overcome in order to scan realistic environments within reasonable coverage times. These immediate problems are as follows: (1) an obstacle avoidance technique which generally maintains the kinematic efficiency of the robot's motion, (2) a means to spread chaotic trajectories across the environment (especially crucial for large and/or complex-shaped environments) that need to be covered, and (3) a real-time coverage calculation technique that is accurate and independent of cell size. This paper aims to progress the field by proposing algorithms that a
    

