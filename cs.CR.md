# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Denoising as a Certified Defense against Clean-label Poisoning](https://arxiv.org/abs/2403.11981) | 提出了一种扩散去噪作为针对干净标签中毒的认证防御，能将攻击成功率降低到0-16%，同时几乎不影响测试准确性，为未来开发更强干净标签攻击和利用该防御措施作为强有力基础提供了重要启示。 |
| [^2] | [Fact-Checking of AI-Generated Reports.](http://arxiv.org/abs/2307.14634) | 本文提出了一种使用相关联的图像对AI生成报告进行事实核查的新方法，以区分报告中的真假句子。这对加快临床工作流程，提高准确性并降低总体成本具有重要意义。 |

# 详细

[^1]: 扩散去噪作为一种针对干净标签中毒的认证防御

    Diffusion Denoising as a Certified Defense against Clean-label Poisoning

    [https://arxiv.org/abs/2403.11981](https://arxiv.org/abs/2403.11981)

    提出了一种扩散去噪作为针对干净标签中毒的认证防御，能将攻击成功率降低到0-16%，同时几乎不影响测试准确性，为未来开发更强干净标签攻击和利用该防御措施作为强有力基础提供了重要启示。

    

    我们提出了一种针对干净标签中毒攻击的认证防御方法。这些攻击通过向训练数据中注入少量的毒害样本（例如1%），其中包含$p$-范数受限的对抗性扰动，从而诱导对测试输入的目标误分类。受到$去噪平滑$实现的对抗鲁棒性的启发，我们展示了如何使用一个现成的扩散模型对篡改的训练数据进行消毒。我们广泛测试了我们的防御措施对七种干净标签中毒攻击的防护效果，并且将它们的攻击成功率降低到0-16%，同时测试准确性几乎没有下降。我们将我们的防御与现有的针对干净标签中毒的对策进行比较，显示出我们的防御效果最佳，并提供最佳的模型效用。我们的结果凸显了未来需要开展更强大的干净标签攻击并使用我们的认证但实用的防御作为稳固基础的必要性。

    arXiv:2403.11981v1 Announce Type: cross  Abstract: We present a certified defense to clean-label poisoning attacks. These attacks work by injecting a small number of poisoning samples (e.g., 1%) that contain $p$-norm bounded adversarial perturbations into the training data to induce a targeted misclassification of a test-time input. Inspired by the adversarial robustness achieved by $denoised$ $smoothing$, we show how an off-the-shelf diffusion model can sanitize the tampered training data. We extensively test our defense against seven clean-label poisoning attacks and reduce their attack success to 0-16% with only a negligible drop in the test time accuracy. We compare our defense with existing countermeasures against clean-label poisoning, showing that the defense reduces the attack success the most and offers the best model utility. Our results highlight the need for future work on developing stronger clean-label attacks and using our certified yet practical defense as a strong base
    
[^2]: AI生成报告的事实核查

    Fact-Checking of AI-Generated Reports. (arXiv:2307.14634v1 [cs.AI])

    [http://arxiv.org/abs/2307.14634](http://arxiv.org/abs/2307.14634)

    本文提出了一种使用相关联的图像对AI生成报告进行事实核查的新方法，以区分报告中的真假句子。这对加快临床工作流程，提高准确性并降低总体成本具有重要意义。

    

    随着生成人工智能（AI）的进步，现在可以生成逼真的自动报告来对放射学图像进行初步阅读。这可以加快临床工作流程，提高准确性并降低总体成本。然而，众所周知，这种模型往往会产生幻觉，导致生成报告中出现错误的发现。在本文中，我们提出了一种使用相关联的图像对AI生成报告进行事实核查的新方法。具体而言，通过学习图像与描述真实或潜在虚假发现的句子之间的关联，开发的核查者区分报告中的真假句子。为了训练这样的核查者，我们首先通过扰动原始与图像相关的放射学报告中的发现，创建了一个新的伪造报告数据集。然后将来自这些报告的真假句子的文本编码与图像编码配对，学习映射到真/假标签的关系。

    With advances in generative artificial intelligence (AI), it is now possible to produce realistic-looking automated reports for preliminary reads of radiology images. This can expedite clinical workflows, improve accuracy and reduce overall costs. However, it is also well-known that such models often hallucinate, leading to false findings in the generated reports. In this paper, we propose a new method of fact-checking of AI-generated reports using their associated images. Specifically, the developed examiner differentiates real and fake sentences in reports by learning the association between an image and sentences describing real or potentially fake findings. To train such an examiner, we first created a new dataset of fake reports by perturbing the findings in the original ground truth radiology reports associated with images. Text encodings of real and fake sentences drawn from these reports are then paired with image encodings to learn the mapping to real/fake labels. The utility 
    

