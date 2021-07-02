
### 1. random search
<!-- - parallel processing toolbox for HP search (ray tune) -->
<!-- https://docs.ray.io/en/master/tune/examples/tune_basic_example.html
https://docs.ray.io/en/master/ray-overview/index.html
https://docs.ray.io/en/master/tune/api_docs/schedulers.html

https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html
https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-cifar.html -->


**Baseline:**  
- blocked: python main.py --gating=None --cuda=False  --ctx_scaling=8 --lrate_sgd=0.01 --centering=False --weight_init=1e-3
- interleaved: python main.py --gating=None --cuda=False  --ctx_scaling=8 --lrate_sgd=0.1 --centering=False --weight_init=1e-3 --training_schedule='interleaved' --ctx_avg=False

**Gated Network, Blocked:**  
python main.py --gating=manual --cuda=True  --ctx_scaling=1 --lrate_sgd=0.01 --centering=False --weight_init=1e-2

**SLA Network, Blocked:**  
- blocked sla, -rew:  python main.py --gating=SLA --cuda=False  --ctx_scaling=2 --lrate_sgd=0.03 --lrate_hebb=0.03 --centering=True --weight_init=1e-3
(if only centering and scaling >>2, prepartitioning sufficient)