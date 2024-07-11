# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Left/Right Brain, human motor control and the implications for robotics.](http://arxiv.org/abs/2401.14057) | 本研究通过训练不同的损失函数，实现了类似于人类的左右半球专门化控制系统，该系统在不同的运动任务中展现出协调性、运动效率和位置稳定性的优势。 |
| [^2] | [Parameter estimation from an Ornstein-Uhlenbeck process with measurement noise.](http://arxiv.org/abs/2305.13498) | 本文研究了带有测量噪声的Ornstein-Uhlenbeck过程参数估计，提出了算法和方法能够分离热噪声和乘性噪声，并改善数据分析的参数估计精度。 |

# 详细

[^1]: 左/右脑、人类运动控制及对机器人的影响

    Left/Right Brain, human motor control and the implications for robotics. (arXiv:2401.14057v1 [cs.RO])

    [http://arxiv.org/abs/2401.14057](http://arxiv.org/abs/2401.14057)

    本研究通过训练不同的损失函数，实现了类似于人类的左右半球专门化控制系统，该系统在不同的运动任务中展现出协调性、运动效率和位置稳定性的优势。

    

    神经网络运动控制器相对传统控制方法具有各种优点，然而由于其无法产生可靠的精确运动，因此尚未得到广泛采用。本研究探讨了一种双侧神经网络架构作为运动任务的控制系统。我们旨在实现类似于人类在不同任务中观察到的半球专门化：优势系统（通常是右手、左半球）擅长协调和运动效率的任务，而非优势系统在需要位置稳定性的任务上表现更好。通过使用不同的损失函数对半球进行训练，实现了专门化。我们比较了具有专门化半球和无专门化半球、具有半球间连接（代表生物学脑桥）和无半球间连接、具有专门化和无专门化的单侧模型。

    Neural Network movement controllers promise a variety of advantages over conventional control methods however they are not widely adopted due to their inability to produce reliably precise movements. This research explores a bilateral neural network architecture as a control system for motor tasks. We aimed to achieve hemispheric specialisation similar to what is observed in humans across different tasks; the dominant system (usually the right hand, left hemisphere) excels at tasks involving coordination and efficiency of movement, and the non-dominant system performs better at tasks requiring positional stability. Specialisation was achieved by training the hemispheres with different loss functions tailored toward the expected behaviour of the respective hemispheres. We compared bilateral models with and without specialised hemispheres, with and without inter-hemispheric connectivity (representing the biological Corpus Callosum), and unilateral models with and without specialisation. 
    
[^2]: 用于带测量噪声的Ornstein-Uhlenbeck过程参数估计

    Parameter estimation from an Ornstein-Uhlenbeck process with measurement noise. (arXiv:2305.13498v1 [stat.ML])

    [http://arxiv.org/abs/2305.13498](http://arxiv.org/abs/2305.13498)

    本文研究了带有测量噪声的Ornstein-Uhlenbeck过程参数估计，提出了算法和方法能够分离热噪声和乘性噪声，并改善数据分析的参数估计精度。

    

    本文旨在研究噪声对Ornstein-Uhlenbeck过程参数拟合的影响，重点考察了乘性噪声和热噪声对信号分离精度的影响。为了解决这些问题，我们提出了有效区分热噪声和乘性噪声、改善参数估计精度的算法和方法，探讨了乘性和热噪声对实际信号混淆的影响，并提出了解决方法。首先，我们提出了一种可以有效分离热噪声的算法，其性能可与Hamilton Monte Carlo (HMC)相媲美，但速度显著提高。随后，我们分析了乘性噪声，并证明了HMC无法隔离热噪声和乘性噪声。然而，我们展示了，在额外了解热噪声和乘性噪声之间比率的情况下，我们可以精确地估计参数和分离信号。

    This article aims to investigate the impact of noise on parameter fitting for an Ornstein-Uhlenbeck process, focusing on the effects of multiplicative and thermal noise on the accuracy of signal separation. To address these issues, we propose algorithms and methods that can effectively distinguish between thermal and multiplicative noise and improve the precision of parameter estimation for optimal data analysis. Specifically, we explore the impact of both multiplicative and thermal noise on the obfuscation of the actual signal and propose methods to resolve them. Firstly, we present an algorithm that can effectively separate thermal noise with comparable performance to Hamilton Monte Carlo (HMC) but with significantly improved speed. Subsequently, we analyze multiplicative noise and demonstrate that HMC is insufficient for isolating thermal and multiplicative noise. However, we show that, with additional knowledge of the ratio between thermal and multiplicative noise, we can accuratel
    

