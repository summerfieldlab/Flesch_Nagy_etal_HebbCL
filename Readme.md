# Code Repository for Flesch, Nagy et al: "Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals"
[![arXiv](https://img.shields.io/badge/arXiv-2203.11560-b31b1b.svg)](https://arxiv.org/abs/2203.11560) [![GitHub license](https://badgen.net/github/license/Summerfieldlab/Flesch_Nagy_etal_HebbCL)](https://github.com/summerfieldlab/Flesch_Nagy_etal_HebbCL/blob/main/LICENSE) [![GitHub issues](https://img.shields.io/github/issues/Summerfieldlab/Flesch_Nagy_etal_HebbCL.svg)](https://GitHub.com/Summerfieldlab/Flesch_Nagy_etal_HebbCL/issues/) [![GitHub pull-requests](https://img.shields.io/github/issues-pr/Summerfieldlab/Flesch_Nagy_etal_HebbCL.svg)](https://GitHub.com/Summerfieldlab/Flesch_Nagy_etal_HebbCL/pull/) 
 [![GitHub forks](https://img.shields.io/github/forks/Summerfieldlab/Flesch_Nagy_etal_HebbCL.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/Summerfieldlab/Flesch_Nagy_etal_HebbCL/network/)
 [![GitHub stars](https://img.shields.io/github/stars/Summerfieldlab/Flesch_Nagy_etal_HebbCL.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/Summerfieldlab/Flesch_Nagy_etal_HebbCL/stargazers/)  

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
```
@article{FleschNagyEtal2022,
  doi = {10.48550/ARXIV.2203.11560},
  
  url = {https://arxiv.org/abs/2203.11560},
  
  author = {Flesch, Timo and Nagy, David G. and Saxe, Andrew and Summerfield, Christopher},
  
  keywords = {Neurons and Cognition (q-bio.NC), Machine Learning (cs.LG), FOS: Biological sciences, FOS: Biological sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals},
  
  publisher = {arXiv},
  
  year = {2022},
  month = {3},
  arxivId = {2203.11560}
  copyright = {Creative Commons Attribution 4.0 International}
}
```