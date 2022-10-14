- ntk_normal.py is run standard SGD with different dimension, width, and C0. The result is shown in Figure 1 and Figure 2 of Section 5.1. You can run as follows:
    python ntk_normal.py --bs 64 --T 1000 --lr 0.1 --num1 0 --num2 1 --width 1000 --size 5 --C0 10 --seed 0
- advtrain.py is run projected adversarial training with different dimension, perturbation size, width, weight deviation. The result is shown in Figure 3 of Section 5.2. You can run as follows:
    python  advtrain.py --bs 32 --T 1000 --lr 0.01 --num1 0 --num2 1 --width 1000 --size 5 --C0 10 --eps 0.2 --atk_iters 100 --seed 0
- advtraingap.py is run projected adversarial training with different width, weight deviation (gap below). The result is shown in Figure 4 of Section 5.2. You can run as follows:
    python advtraingap.py --T 1000 --size 28 --width 1000 --lr 0.01 --eps 0.2 --gap 0.1 --seed 0
