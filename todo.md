### todo list 

### logger 
<!-- - log performance, network weights, layer-wise activity patterns -->

### code refactoring
- Optimiser into optimiser.py
- train_model into utils.trainer.py ?
- checkpoint folders into logger/init
- logger inheritance: loggerBase, loggerMLP, loggerCNN ?

### random search
<!-- - parallel processing toolbox for HP search (ray tune) -->
<!-- https://docs.ray.io/en/master/tune/examples/tune_basic_example.html
https://docs.ray.io/en/master/ray-overview/index.html
https://docs.ray.io/en/master/tune/api_docs/schedulers.html

https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html
https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-cifar.html -->


### fits to human data
- train at single trial level

### analysis notebook: model performance and rsa
- notebook with analysis scripts to replicate cosyne poster
- simulation 1: vanilla blocked vs interleaved
- simulation 2: gated blocked 
- simulation 3: only GHA/SLA
- simulation 4: SLA network blocked
- simulation 5: sluggish sla interleaved
- simulation 6: sluggish sla on human choices 

### analysis notebook: evaluate model fits
    - plot model predictions (average over cross-validated participants) alongside participant data
        - accuracy
        - sigmoids (rel/irrel)
        - choice matrices
        - rsa on human data and model's choices
        - choice model on human data and model's choices
