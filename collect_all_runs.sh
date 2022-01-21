#!/bin/bash
#collect individual runs


# # blocked, baseline
python collect_runs.py --gating=None --cuda=False  --ctx_scaling=4 --lrate_sgd=0.03 --lrate_hebb=0.009 --centering=False --weight_init=1e-2 --save_dir='baseline_blocked_new' --verbose=False;

# interleaved, baseline
python collect_runs.py --gating=None --cuda=False  --ctx_scaling=5 --lrate_sgd=0.2 --centering=False --weight_init=1e-2 --training_schedule='interleaved' --ctx_avg=False --save_dir='baseline_interleaved_new' --verbose=False;

# blocked, gated 
python collect_runs.py --gating=manual --cuda=False  --ctx_scaling=1 --lrate_sgd=0.01 --centering=False --weight_init=1e-2 --save_dir='gated_blocked_new' --verbose=False;

# sla, gated
python collect_runs.py --gating=SLA --cuda=False  --ctx_scaling=5 --lrate_sgd=0.03 --lrate_hebb=0.009 --centering=True --weight_init=1e-2 --save_dir='sla_blocked_new' --verbose=False;