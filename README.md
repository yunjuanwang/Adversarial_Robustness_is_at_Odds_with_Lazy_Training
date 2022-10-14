# Adversarial Robustness is at Odds with Lazy Training

[NeurIPS2022] A PyTorch implementation for the experiments in the paper.

## Abstract

Recent works show that random neural networks are vulnerable against adversarial attacks [Daniely and Schacham, 2020] and that such attacks can be easily found using a single step of gradient descent [Bubeck et al., 2021]. In this work, we take one step further and show that one single gradient step can find adversarial examples for networks trained in the so-called lazy regime. This regime is interesting because even though the neural network weights remain close to the initialization, there exist networks with small generalization error, which can be found efficiently using first-order methods. Our work challenges the model of the lazy regime, the only regime in which neural networks are provably efficiently learnable. We show that the networks trained in this regime, even though they enjoy good theoretical computational guarantees, remain vulnerable to adversarial examples. In doing so, we resolve an open question posed by the work of [Bubeck et al., 2021], who show a similar result for random neural networks. To the best of our knowledge, this is the first work to prove that such well-generalizable neural networks are still vulnerable to adversarial attacks. 

## Implementation

1. ntk_normal.py is run standard SGD with different dimension, width, and C0. The result is shown in Figure 1 and Figure 2 of Section 5.1. You can run as follows:
    python ntk_normal.py --bs 64 --T 1000 --lr 0.1 --num1 0 --num2 1 --width 1000 --size 5 --C0 10 --seed 0
2. advtrain.py is run projected adversarial training with different dimension, perturbation size, width, weight deviation. The result is shown in Figure 3 of Section 5.2. You can run as follows:
    python  advtrain.py --bs 32 --T 1000 --lr 0.01 --num1 0 --num2 1 --width 1000 --size 5 --C0 10 --eps 0.2 --atk_iters 100 --seed 0
3. advtraingap.py is run projected adversarial training with different width, weight deviation (gap below). The result is shown in Figure 4 of Section 5.2. You can run as follows:
    python advtraingap.py --T 1000 --size 28 --width 1000 --lr 0.01 --eps 0.2 --gap 0.1 --seed 0
