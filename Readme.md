# TinyMoE

This models pushes performance of small models as far as possible. We aim to create a Model that is both large and small at the same time. Large to take advantage of large scale pretraining and large amounts of GPU memory available. But small to be able to serve lighting fast inference.

To accomplish this we take inspiration of a few recent models(Mixtral and Deepseek-MoE), mainly:

- MoE (Mixture of Experts work to increase model size without utilizing all params for inference) [0](https://huggingface.co/blog/moe#when-to-use-sparse-moes-vs-dense-models)
- Grouped Query Attention (downscaled KV keys to increase attention effeceincy)[1](https://arxiv.org/abs/2305.13245v3)
- Expert Specialization(More effecient experts)[2](https://arxiv.org/pdf/2401.06066.pdf)
- Per layer Configuration of Sliding Window Attention and Grouped Query Attention Sizes, We use lot's of early layers with smaller windows and attention head counts for speed, and a few layers of denser global attention

We aim for 440M active parameters and 5B trainable. Ideally runs at GPT-2-medium level speeds for inference.

Currently
- [x] Model Architecture Done
- [ ] Tuning the model architeture hyper parameters for inference speed
- [x] Train Simple Variants on effecient web and synthetic data
- [ ] Training Model on 1T+ tokens
