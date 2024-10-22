# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models](https://arxiv.org/abs/2402.08151) | 本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。 |
| [^2] | [Privacy-Preserving Community Detection for Locally Distributed Multiple Networks.](http://arxiv.org/abs/2306.15709) | 本文提出了一种保护隐私的本地分布多网络社区检测方法，利用隐私保护来进行共识社区检测和估计。采用随机响应机制对网络边进行扰动，通过隐私保护分布式谱聚类算法在扰动邻接矩阵上执行，以防止社区之间的抵消。同时，开发了两步偏差调整过程来消除扰动和网络矩阵带来的偏差。 |

# 详细

[^1]: 渐变流自适应重要性抽样用于sigmoid分类模型的贝叶斯留一交叉验证

    Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models

    [https://arxiv.org/abs/2402.08151](https://arxiv.org/abs/2402.08151)

    本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。

    

    我们引入了一组梯度流引导的自适应重要性抽样（IS）变换，用于稳定贝叶斯分类模型的点级留一交叉验证（LOO）预测的蒙特卡罗近似。可以利用这种方法来评估模型的普适性，例如计算与AIC类似的LOO或计算LOO ROC / PRC曲线以及派生的度量指标，如AUROC和AUPRC。通过变分法和梯度流，我们推导出两个简单的非线性单步变换，利用梯度信息将模型的预训练完整数据后验靠近目标LOO后验预测分布。这样，变换稳定了重要性权重。因为变换涉及到似然函数的梯度，所以结果的蒙特卡罗积分依赖于模型Hessian的Jacobian行列式。我们推导出了这些Jacobian行列式的闭合精确公式。

    We introduce a set of gradient-flow-guided adaptive importance sampling (IS) transformations to stabilize Monte-Carlo approximations of point-wise leave one out cross-validated (LOO) predictions for Bayesian classification models. One can leverage this methodology for assessing model generalizability by for instance computing a LOO analogue to the AIC or computing LOO ROC/PRC curves and derived metrics like the AUROC and AUPRC. By the calculus of variations and gradient flow, we derive two simple nonlinear single-step transformations that utilize gradient information to shift a model's pre-trained full-data posterior closer to the target LOO posterior predictive distributions. In doing so, the transformations stabilize importance weights. Because the transformations involve the gradient of the likelihood function, the resulting Monte Carlo integral depends on Jacobian determinants with respect to the model Hessian. We derive closed-form exact formulae for these Jacobian determinants in
    
[^2]: 保护隐私的本地分布多网络社区检测

    Privacy-Preserving Community Detection for Locally Distributed Multiple Networks. (arXiv:2306.15709v1 [cs.SI])

    [http://arxiv.org/abs/2306.15709](http://arxiv.org/abs/2306.15709)

    本文提出了一种保护隐私的本地分布多网络社区检测方法，利用隐私保护来进行共识社区检测和估计。采用随机响应机制对网络边进行扰动，通过隐私保护分布式谱聚类算法在扰动邻接矩阵上执行，以防止社区之间的抵消。同时，开发了两步偏差调整过程来消除扰动和网络矩阵带来的偏差。

    

    现代多层网络由于隐私、所有权和通信成本的原因，常常以本地和分布式的方式存储和分析。关于基于这些数据的模型化统计方法用于社区检测的文献仍然有限。本文提出了一种新的方法，用于基于本地存储和计算的网络数据的多层随机块模型中的共识社区检测和估计，并采用隐私保护。开发了一种名为隐私保护分布式谱聚类（ppDSC）的新算法。为了保护边的隐私，我们采用了随机响应（RR）机制来扰动网络边，该机制满足差分隐私的强概念。ppDSC算法在平方的RR扰动邻接矩阵上执行，以防止不同层之间的社区相互抵消。为了消除RR和平方网络矩阵所带来的偏差，我们开发了一个两步偏差调整过程。

    Modern multi-layer networks are commonly stored and analyzed in a local and distributed fashion because of the privacy, ownership, and communication costs. The literature on the model-based statistical methods for community detection based on these data is still limited. This paper proposes a new method for consensus community detection and estimation in a multi-layer stochastic block model using locally stored and computed network data with privacy protection. A novel algorithm named privacy-preserving Distributed Spectral Clustering (ppDSC) is developed. To preserve the edges' privacy, we adopt the randomized response (RR) mechanism to perturb the network edges, which satisfies the strong notion of differential privacy. The ppDSC algorithm is performed on the squared RR-perturbed adjacency matrices to prevent possible cancellation of communities among different layers. To remove the bias incurred by RR and the squared network matrices, we develop a two-step bias-adjustment procedure.
    

