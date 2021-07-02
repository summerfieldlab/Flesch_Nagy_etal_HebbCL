### todo list 
1. collect individual runs 
2. new branch: fit to human choice patterns 
    - import human choice patterns 
    - new dataset generator (takes choice mats as input)
    - new loss function & model
    - trial run with interleaved + sluggishness 
    - raytune: find best hyperparameters for each choicemat (single participants or at grp level)
    - collect multiple runs with best fitting parameters 
### logger 
<!-- - log performance, network weights, layer-wise activity patterns -->

### code refactoring
- Optimiser into optimiser.py
- train_model into utils.trainer.py ?
- checkpoint folders into logger/init
- logger inheritance: loggerBase, loggerMLP, loggerCNN ?

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

### 2. collect individual runs (parpool)
- for each simu above, collect ~30 independent runs (parallel processing toolbox)

### 3. analyse data 
### fits to human data
- train at single trial level

### analysis notebook: model performance and rsa
- notebook with analysis scripts to replicate cosyne poster
- simulation 1: vanilla blocked vs interleaved
- simulation 2: gated blocked 
- simulation 3: only GHA/SLA
- simulation 4: SLA network blocked
- simulation 5: sluggish sla interleaved: show choices/accuracy for different levels of sluggishness
- simulation 6: sluggish sla on human choices 

### analysis notebook: evaluate model fits
    - plot model predictions (average over cross-validated participants) alongside participant data
        - accuracy
        - sigmoids (rel/irrel)
        - choice matrices
        - rsa on human data and model's choices
        - choice model on human data and model's choices
