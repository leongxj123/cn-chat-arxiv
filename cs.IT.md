# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Approaching Rate-Distortion Limits in Neural Compression with Lattice Transform Coding](https://arxiv.org/abs/2403.07320) | 格点变换编码（LTC）通过在潜空间中采用格点量化，实现了神经压缩中接近速率失真极限的优化。 |

# 详细

[^1]: 用格点变换编码接近神经压缩中的速率失真极限

    Approaching Rate-Distortion Limits in Neural Compression with Lattice Transform Coding

    [https://arxiv.org/abs/2403.07320](https://arxiv.org/abs/2403.07320)

    格点变换编码（LTC）通过在潜空间中采用格点量化，实现了神经压缩中接近速率失真极限的优化。

    

    神经压缩在设计具有良好速率失真（RD）性能但复杂度低的有损压缩器方面取得了巨大进展。迄今为止，神经压缩设计涉及将源转换为潜变量，然后舍入为整数并进行熵编码。尽管这种方法已被证明在某些源上的一次性情况下是最佳的，但我们表明在i.i.d.序列上它是高度次优的，事实上总是恢复原始源序列的标量量化。我们展示亚优越性是由于潜空间中量化方案的选择，而非变换设计所致。通过在潜空间中采用格点量化而非标量量化，我们展示了格点变换编码（Lattice Transform Coding，LTC）能够在各个维度上恢复最佳矢量量化，并在合理的复杂度下接近渐近可实现的速率失真函数。

    arXiv:2403.07320v1 Announce Type: cross  Abstract: Neural compression has brought tremendous progress in designing lossy compressors with good rate-distortion (RD) performance at low complexity. Thus far, neural compression design involves transforming the source to a latent vector, which is then rounded to integers and entropy coded. While this approach has been shown to be optimal in a one-shot sense on certain sources, we show that it is highly sub-optimal on i.i.d. sequences, and in fact always recovers scalar quantization of the original source sequence. We demonstrate that the sub-optimality is due to the choice of quantization scheme in the latent space, and not the transform design. By employing lattice quantization instead of scalar quantization in the latent space, we demonstrate that Lattice Transform Coding (LTC) is able to recover optimal vector quantization at various dimensions and approach the asymptotically-achievable rate-distortion function at reasonable complexity. 
    

