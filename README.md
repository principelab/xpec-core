# xPEC

**xPEC** is a tool for automatic detection of data models from neural activity data, guided purely by the structure of the data.
<br>
<br>
xPEC leverages **prediction error connectivity (PEC)** as a network marker, which relates to the complexity of information contained in the network and its consistency across repetitions ​([Principe et al. 2019](https://doi.org/10.1016/j.neuroimage.2018.11.052))​.
<br>
<br>
<p align="center">
  <img src="data/figures/simulation/FIG2A.jpg" alt="System Diagram" width="500">
</p>

## Quick Start

```bash
git clone https://github.com/ivkarla/xpec-core
cd xpec-core
conda create -n xpec python=3.10
conda activate xpec
pip install -e .
