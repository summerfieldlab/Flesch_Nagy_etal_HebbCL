### Best fitting parameters

#### vanilla blocked vs interleaved
- blocked: 
- interleaved: 

- blocked sla, mse: hebb norm 1, sgd lrate 1e-3
- blocked sla, -rew: hebb norm 10, sgd lrate 1e-2


### manually gated net:
python main.py --gating=manual --cuda=True  --ctx_scaling=1 --lrate_sgd=0.01 --centering=False --weight_init=1e-2