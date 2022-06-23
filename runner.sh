#!/bin/bash
#collect individual runs


# # # blocked, baseline
# python collect_runs.py --gating=None --cuda=False  --ctx_scaling=4 --lrate_sgd=0.03 --lrate_hebb=0.009 --centering=False --weight_init=1e-2 --save_dir=baseline_blocked_new_select --verbose=False;

# # interleaved, baseline
# python collect_runs.py --gating=None --cuda=False  --training_schedule=interleaved --ctx_scaling=5 --lrate_sgd=0.2 --centering=False --weight_init=1e-2  --ctx_avg=False --save_dir=baseline_interleaved_new_select --verbose=False;

# # blocked, gated 
# python collect_runs.py --gating=manual --cuda=False  --ctx_scaling=1 --lrate_sgd=0.01 --centering=True --weight_init=1e-2 --save_dir=gated_blocked_new_select_cent --verbose=False;

# # oja, gated only context units (as in arxiv preprint)
# python collect_runs.py --gating=oja_ctx --cuda=False --save_dir=oja_blocked_new_select_halfcenter --ctx_scaling=1 --lrate_sgd=0.03 --lrate_hebb=0.05 --centering=True --weight_init=1e-2  --verbose=False


# ----------------------------
# revisions: 


# oja blocked, 8 episodes only 
python collect_runs.py --gating=oja_ctx --cuda=False --save_dir=oja_ctx_8episodes_blocked --ctx_scaling=3 --lrate_sgd=0.09207067771676251 --lrate_hebb=0.0039883754510576805 --centering=True --weight_init=1e-2  --verbose=False --n_episodes=8

# oja interleaved, 8 episodes only 
python collect_runs.py --gating=oja_ctx --cuda=False --save_dir=oja_ctx_8episodes_interleaved --training_schedule=interleaved --ctx_scaling=3 --lrate_sgd=0.08710014100174149 --lrate_hebb=0.005814333717889643 --centering=True --weight_init=1e-2  --verbose=False --n_episodes=8

# above with different levels of sluggishness:
python sluggish_runs.py