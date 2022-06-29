# Code Repository for Flesch, Nagy et al: "Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals"
The repo is work in progress, stay tuned!


## Usage
To replicate results reported in the paper, clone this repository and install the required packages (preferably in a separate environment):
```bash
pip install -r requirements.txt
```
To re-run all simulations and collect several independent training runs, open a command window and run the following bash script:
```bash
./runner.sh
```

For individual runs, you can call the `main.py` file with command line arguments.  
If you want to run your own hyperparameter optimisation, have a look at the `HPOTuner` class in `hebbcl.tuner`.  

To replicate analyses and create figures, have a look at the `paper_figures_scratchpad.ipynb` notebook in the `notebooks` subfolder.
## Preprint
For a preprint of this work, see [https://arxiv.org/abs/2203.11560](https://arxiv.org/abs/2203.11560)

## Citation
If you'd like to cite this work, please use the following format:
```@misc{https://doi.org/10.48550/arxiv.2203.11560,
  doi = {10.48550/ARXIV.2203.11560},
  
  url = {https://arxiv.org/abs/2203.11560},
  
  author = {Flesch, Timo and Nagy, David G. and Saxe, Andrew and Summerfield, Christopher},
  
  keywords = {Neurons and Cognition (q-bio.NC), Machine Learning (cs.LG), FOS: Biological sciences, FOS: Biological sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```