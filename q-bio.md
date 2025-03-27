# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors.](http://arxiv.org/abs/2401.02739) | 本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。 |

# 详细

[^1]: 扩散变分推断：扩散模型作为表达性变分后验

    Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors. (arXiv:2401.02739v1 [cs.LG])

    [http://arxiv.org/abs/2401.02739](http://arxiv.org/abs/2401.02739)

    本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。

    

    我们提出了去噪扩散变分推断（DDVI），一种用扩散模型作为表达性变分后验的潜变量模型的近似推断算法。我们的方法通过辅助潜变量增加了变分后验，从而得到一个表达性的模型类，通过反转用户指定的加噪过程在潜空间中进行扩散。我们通过优化一个受到觉醒-睡眠算法启发的边际似然新下界来拟合这些模型。我们的方法易于实现（它适配了正则化的ELBO扩展），与黑盒变分推断兼容，并且表现优于基于归一化流或对抗网络的替代近似后验类别。将我们的方法应用于深度潜变量模型时，我们的方法得到了去噪扩散变分自动编码器（DD-VAE）算法。我们将该算法应用于生物学中的一个激励任务 -- 从人类基因组中推断潜在血统 -- 超过了强基线模型。

    We propose denoising diffusion variational inference (DDVI), an approximate inference algorithm for latent variable models which relies on diffusion models as expressive variational posteriors. Our method augments variational posteriors with auxiliary latents, which yields an expressive class of models that perform diffusion in latent space by reversing a user-specified noising process. We fit these models by optimizing a novel lower bound on the marginal likelihood inspired by the wake-sleep algorithm. Our method is easy to implement (it fits a regularized extension of the ELBO), is compatible with black-box variational inference, and outperforms alternative classes of approximate posteriors based on normalizing flows or adversarial networks. When applied to deep latent variable models, our method yields the denoising diffusion VAE (DD-VAE) algorithm. We use this algorithm on a motivating task in biology -- inferring latent ancestry from human genomes -- outperforming strong baselines
    

