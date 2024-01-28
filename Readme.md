# TinyMoE

This models pushes performance of small models as far as possible. The field of small models has been largely ignored in the rush to scale. We aim to create a Model that is both large and small at the same time. Large to take advantage of large scale pretraining and large amounts of GPU memory available. But small to be able to serve lighting fast inference.

To accomplish this we take inspiration of a few recent models(Mixtral and Deepseek-MoE), chiefly:

- MoE (Mixture of Experts work to increase model size without utilizing all params for inference) [0](https://huggingface.co/blog/moe#when-to-use-sparse-moes-vs-dense-models)
- Grouped Query Attention (downscaled KV keys to increase attention effeceincy)[1](https://arxiv.org/abs/2305.13245v3)
- Expert Specialization(More effecient experts)[2](https://arxiv.org/pdf/2401.06066.pdf)

We also shift the parameter distribution to a uniquely small number of active parameters. And a high ratio of trainable:inference params

We aim for 440M active parameters and 5B trainable 

