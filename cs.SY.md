# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [State space representations of the Roesser type for convolutional layers](https://arxiv.org/abs/2403.11938) | 从控制理论的角度，提供了Roesser类型的2-D卷积层状态空间表示，具有最小化的状态数量，在$c_\mathrm{in}=c_\mathrm{out}$的情况下证明了这一点，并进一步实现了扩张、跨越和N-D卷积的状态空间表示。 |
| [^2] | [Deep Learning Safety Concerns in Automated Driving Perception.](http://arxiv.org/abs/2309.03774) | 本研究旨在通过引入安全考虑作为结构元素，以系统综合的方式确保基于深度神经网络的自动驾驶系统的安全性。这一概念不仅与现有的安全标准相契合，还为AI安全相关的学术出版物和标准提供了新的启示。 |

# 详细

[^1]: Roesser类型的状态空间表示用于卷积层

    State space representations of the Roesser type for convolutional layers

    [https://arxiv.org/abs/2403.11938](https://arxiv.org/abs/2403.11938)

    从控制理论的角度，提供了Roesser类型的2-D卷积层状态空间表示，具有最小化的状态数量，在$c_\mathrm{in}=c_\mathrm{out}$的情况下证明了这一点，并进一步实现了扩张、跨越和N-D卷积的状态空间表示。

    

    从控制理论的角度看，卷积层（神经网络的）是2-D（或N-D）线性时不变动态系统。卷积层通常通过卷积核表示，对应于动态系统通过其脉冲响应表示。然而，许多控制理论的分析工具，例如涉及线性矩阵不等式的工具，需要一个状态空间表示。因此，我们明确提供了Roesser类型的2-D卷积层状态空间表示，具有$c_\mathrm{in}r_1+c_\mathrm{out}r_2$个状态，其中$c_\mathrm{in}/c_\mathrm{out}$是层的输入/输出通道数，$r_1/r_2$ 表示卷积核的宽度/长度。对于$c_\mathrm{in}=c_\mathrm{out}$，已经证明这种表示是最小的。我们进一步构建了扩张、跨越和N-D卷积的状态空间表示。

    arXiv:2403.11938v1 Announce Type: cross  Abstract: From the perspective of control theory, convolutional layers (of neural networks) are 2-D (or N-D) linear time-invariant dynamical systems. The usual representation of convolutional layers by the convolution kernel corresponds to the representation of a dynamical system by its impulse response. However, many analysis tools from control theory, e.g., involving linear matrix inequalities, require a state space representation. For this reason, we explicitly provide a state space representation of the Roesser type for 2-D convolutional layers with $c_\mathrm{in}r_1 + c_\mathrm{out}r_2$ states, where $c_\mathrm{in}$/$c_\mathrm{out}$ is the number of input/output channels of the layer and $r_1$/$r_2$ characterizes the width/length of the convolution kernel. This representation is shown to be minimal for $c_\mathrm{in} = c_\mathrm{out}$. We further construct state space representations for dilated, strided, and N-D convolutions.
    
[^2]: 自动驾驶感知中的深度学习安全考虑

    Deep Learning Safety Concerns in Automated Driving Perception. (arXiv:2309.03774v1 [cs.LG])

    [http://arxiv.org/abs/2309.03774](http://arxiv.org/abs/2309.03774)

    本研究旨在通过引入安全考虑作为结构元素，以系统综合的方式确保基于深度神经网络的自动驾驶系统的安全性。这一概念不仅与现有的安全标准相契合，还为AI安全相关的学术出版物和标准提供了新的启示。

    

    深度学习领域的最新进展以及深度神经网络（DNNs）在感知方面的出色性能导致了对其在自动驾驶系统中应用的增加需求。这类系统的安全性至关重要，因此需要考虑DNNs的独特属性。为了以系统综合的方式确保基于DNNs的自动驾驶系统的安全性，引入了所谓的安全考虑作为适当的结构元素。一方面，安全考虑的概念设计与现有的与自动驾驶系统安全相关的标准如ISO 21448（SOTIF）非常契合。另一方面，它已经激发了几篇学术出版物和即将出台的关于AI安全的标准，如ISO PAS 8800。虽然安全考虑的概念以前已经被介绍过，但本文对其进行了扩展和优化，借鉴了各个领域和安全专家的反馈意见。

    Recent advances in the field of deep learning and impressive performance of deep neural networks (DNNs) for perception have resulted in an increased demand for their use in automated driving (AD) systems. The safety of such systems is of utmost importance and thus requires to consider the unique properties of DNNs.  In order to achieve safety of AD systems with DNN-based perception components in a systematic and comprehensive approach, so-called safety concerns have been introduced as a suitable structuring element. On the one hand, the concept of safety concerns is -- by design -- well aligned to existing standards relevant for safety of AD systems such as ISO 21448 (SOTIF). On the other hand, it has already inspired several academic publications and upcoming standards on AI safety such as ISO PAS 8800.  While the concept of safety concerns has been previously introduced, this paper extends and refines it, leveraging feedback from various domain and safety experts in the field. In par
    

