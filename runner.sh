#!/bin/bash
#collect individual runs

# ---- preprint ----
# blocked, baseline
python collect_runs.py --gating=None --cuda=False  --ctx_scaling=4 --lrate_sgd=0.03 --lrate_hebb=0.009 --centering=False --weight_init=1e-2 --save_dir=baseline_blocked_new_select --verbose=False;

# interleaved, baseline
python collect_runs.py --gating=None --cuda=False  --training_schedule=interleaved --ctx_scaling=5 --lrate_sgd=0.2 --centering=False --weight_init=1e-2  --ctx_avg=False --save_dir=baseline_interleaved_new_select --verbose=False;

# blocked, gated 
python collect_runs.py --gating=manual --cuda=False  --ctx_scaling=1 --lrate_sgd=0.01 --centering=True --weight_init=1e-2 --save_dir=gated_blocked_new_select_cent --verbose=False;

# oja, gated only context units (as in arxiv preprint)
python collect_runs.py --gating=oja_ctx --cuda=False --save_dir=oja_blocked_new_select_halfcenter --ctx_scaling=1 --lrate_sgd=0.03 --lrate_hebb=0.05 --centering=True --weight_init=1e-2  --verbose=False


# ---- revisions ----
# oja blocked, 200 episodes 
python collect_runs.py --gating=oja --cuda=False --save_dir=blobs_revision_200episodes_blocked_oja --ctx_scaling=2 --lrate_sgd=0.03775369549108046 --lrate_hebb=0.00021666673995458582 --centering=True --weight_init=1e-2  --verbose=False --n_episodes=200 --perform_hebb=True --ctx_avg=False

# oja interleaved, 200 episodes 
python collect_runs.py --gating=oja --cuda=False --save_dir=blobs_revision_200episodes_interleaved_oja --training_schedule=interleaved --ctx_scaling=1 --lrate_sgd=0.07638775393909703 --lrate_hebb=0.000269741524659589 --centering=True --weight_init=1e-2  --verbose=False --n_episodes=8 --perform_hebb=True --ctx_avg=False


# vanilla blocked, 8 episodes only
python collect_runs.py --gating=None --cuda=False --save_dir=blobs_revision_8episodes_blocked_vanilla --ctx_scaling=2 --lrate_sgd=0.2 --centering=False --weight_init=1e-2 --verbose=False --n_episodes=8 --perform_hebb=False --ctx_avg=False

# vanilla interleaved, 8 episodes only
python collect_runs.py --gating=None --cuda=False --save_dir=blobs_revision_8episodes_interleaved_vanilla --training_schedule=interleaved --ctx_scaling=6 --lrate_sgd=0.0981598358572433 --centering=True --weight_init=1e-2 --verbose=False --n_episodes=8 --perform_hebb=False --ctx_avg=False

# oja blocked, 8 episodes only 
python collect_runs.py --gating=oja --cuda=False --save_dir=blobs_revision_8episodes_blocked_oja --ctx_scaling=3 --lrate_sgd=0.09056499086887726 --lrate_hebb=0.0025838610435258585 --centering=True --weight_init=1e-2  --verbose=False --n_episodes=8 --perform_hebb=True --ctx_avg=False

# oja interleaved, 8 episodes only 
python collect_runs.py --gating=oja --cuda=False --save_dir=blobs_revision_8episodes_interleaved_oja --training_schedule=interleaved --ctx_scaling=4 --lrate_sgd=0.09263634569936459 --lrate_hebb=0.0003276905554752727 --centering=True --weight_init=1e-2  --verbose=False --n_episodes=8 --perform_hebb=True --ctx_avg=False

# # above with different levels of sluggishness:
# python sluggish_runs.py

# trees vanilla interleaved 
python collect_runs_trees.py --cuda=False --verbose=False --training_schedule=interleaved --ctx_avg=False --centering=True --ctx_scaling=1 --n_episodes=100 --perform_hebb=False --gating=None --lrate_sgd=0.0018549176154984076 --save_dir=trees_asha_interleaved_vanilla_1ctx

# trees vanilla blocked 
python collect_runs_trees.py --cuda=False --verbose=False --ctx_avg=False --centering=True --ctx_scaling=4 --n_episodes=100 --perform_hebb=False --gating=None --lrate_sgd=0.00196874872857594 --save_dir=trees_asha_blocked_vanilla_1ctx

# trees oja interleaved
python collect_runs_trees.py --cuda=False --verbose=False --training_schedule=interleaved --ctx_avg=False --centering=True --ctx_scaling=1 --n_episodes=100 --perform_hebb=True --gating=oja_ctx --lrate_sgd=0.0018549176154984076 --lrate_hebb=0.0066835760364487365 --save_dir=trees_asha_interleaved_ojactx_1ctx

# trees oja blocked
python collect_runs_trees.py --cuda=False --verbose=False --ctx_avg=False --centering=True --ctx_scaling=4 --n_episodes=100 --perform_hebb=True --gating=oja_ctx --lrate_sgd=0.00196874872857594 --lrate_hebb=0.0008495631690508217 --save_dir=trees_asha_blocked_ojactx_1ctx