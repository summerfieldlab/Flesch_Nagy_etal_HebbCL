### todo list 
1. collect individual runs 
2. new branch: sluggishness 
    <!-- - simulate interleaved network for range of sluggishness parameters (several runs per sluggishness param) -->
    <!-- - gridsearch: find best hyperparameters for each choicemat (single participants or at grp level) -->
    - look at results of sluggishness simu
    - sluggishness: collect multiple runs with best fitting parameters 
3. analysis notebook
    - import runs, create alldata dict 
    - tbc

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




-----------------



### code refactoring
- Optimiser into optimiser.py
- train_model into utils.trainer.py ?
- logger inheritance: loggerBase, loggerMLP, loggerCNN ?

