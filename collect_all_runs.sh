#!/bin/bash
#collect individual runs


# blocked, baseline
python collect_runs.py --gating=None --cuda=False  --ctx_scaling=8 --lrate_sgd=0.01 --centering=False --weight_init=1e-3 --save_dir='baseline_blocked' --verbose=False;

# interleaved, baseline
python collect_runs.py --gating=None --cuda=False  --ctx_scaling=8 --lrate_sgd=0.1 --centering=False --weight_init=1e-3 --training_schedule='interleaved' --ctx_avg=False --save_dir='baseline_interleaved' --verbose=False;

# blocked, gated 
python collect_runs.py --gating=manual --cuda=True  --ctx_scaling=1 --lrate_sgd=0.01 --centering=False --weight_init=1e-2 --save_dir='gated_blocked' --verbose=False;

# sla, gated
python collect_runs.py --gating=SLA --cuda=False  --ctx_scaling=2 --lrate_sgd=0.03 --lrate_hebb=0.03 --centering=True --weight_init=1e-3 --save_dir='sla_blocked' --verbose=False;