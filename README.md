# bigthree-dynet-pd

Simulation code for the article:

> Abián, D., Bernad, J., Ilarri, S. *et al*. Individual and collective gains from cooperation and reciprocity in a dynamic-network Prisoner’s Dilemma driven by extraversion, openness, and agreeableness. *Scientific Reports* (2026). https://doi.org/10.1038/s41598-026-49942-w

This repository implements an agent-based simulation of a dynamic-network Prisoner’s Dilemma in which three continuous personality traits (Extraversion, Openness, Agreeableness) shape:

- agents’ ideal number of concurrent partners (Extraversion),
- the breadth of partner search (Openness),
- and baseline cooperativeness (Agreeableness),

with local, history-dependent cooperation and personality-driven tie formation and cutting.

The code reproduces exactly the simulations reported in the article, including the grid over group sizes, trait scenarios, and trait–history mixing weights.

---

## 🔔 Citation (please read before using)

If you use this code or any datasets generated with it in scientific, academic, or technical work, please cite the primary article. When relevant, also cite the software repository and the archived dataset.

### Primary article (main citation)

> Abián, D., Bernad, J., Ilarri, S. *et al*. Individual and collective gains from cooperation and reciprocity in a dynamic-network Prisoner’s Dilemma driven by extraversion, openness, and agreeableness. *Scientific Reports* (2026). https://doi.org/10.1038/s41598-026-49942-w

### Software repository and dataset record

> Abián, D., Bernad, J., Ilarri, S., & Trillo-Lado, R. (2025).  
> *bigthree-dynet-pd: Dynamic-network Prisoner’s Dilemma simulation with personality-driven tie dynamics* [Computer software].  
> GitHub. https://github.com/davidabian/bigthree-dynet-pd

> Abián, D., Bernad, J., Ilarri, S., & Trillo-Lado, R. (2025).  
> *Simulation outputs for “Individual and collective gains from cooperation and reciprocity in a dynamic-network Prisoner’s Dilemma driven by extraversion, openness, and agreeableness”* [Data set].  
> Zenodo. https://doi.org/10.5281/zenodo.17714612

### BibTeX entry

```bibtex
@article{Abian2026DynamicPD,
  author  = {Abi{\'a}n, David and Bernad, Jorge and Ilarri, Sergio and Trillo-Lado, Raquel},
  title   = {Individual and collective gains from cooperation and reciprocity in a dynamic-network Prisoner's Dilemma driven by extraversion, openness, and agreeableness},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-49942-w},
  url     = {https://doi.org/10.1038/s41598-026-49942-w},
}
````

---

## 🚀 Installation

### Using `pip` and a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Using `conda`

```bash
conda env create -f environment.yml
conda activate bigthree-dynet-pd
```

The main dependencies are:

* `numpy >= 2.0.2`
* `pandas >= 2.2.3`
* `networkx >= 3.2.1`
* `scipy >= 1.13.1`

---

## ▶️ Quick start

### 1. Small demo run

To run a small demonstration grid (few turns and seeds, for testing):

```bash
chmod +x examples/run_small_demo.sh
./examples/run_small_demo.sh
```

This will create a directory called `grid_small_demo/` with:

* per-run subdirectories containing CSV outputs,
* an overall `grid_runs_summary.csv`,
* and aggregated statistics under `grid_small_demo/aggregate/`.

### 2. Full grid used in the article

To reproduce the full grid used in the article:

```bash
chmod +x run_simulations.sh
./run_simulations.sh
```

This will create a directory called `grid_bigthree_dynet_pd/` with:

* one subdirectory per combination of parameters (`run_tag`),
* per-run CSV files:

  * `agents.csv`
  * `agent_timeseries.csv`
  * `correlations.csv`
  * `assortativity.csv`
  * `network_stats.csv`
  * `parameters.json`
* a top-level `grid_runs_summary.csv`,
* and aggregated results in `grid_bigthree_dynet_pd/aggregate/`, which are the inputs for the figures and analyses reported in the paper.

---

## 📁 Repository structure

```text
bigthree-dynet-pd/
├─ README.md                 # This file
├─ LICENSE                   # Open-source license (MIT)
├─ CITATION.cff              # Machine-readable citation info
├─ requirements.txt          # Python dependencies (pip)
├─ environment.yml           # Conda environment (optional)
├─ .gitignore
├─ bigthree_dynet_pd.py      # Main simulation script (grid-capable CLI)
├─ run_simulations.sh        # Full grid configuration used in the article
├─ examples/
│   └─ run_small_demo.sh     # Small demo grid for testing
└─ README_dataset.md         # Documentation for the ZIP dataset (for Zenodo/Zaguán)
```

---

## 📊 Data availability

The full set of simulation outputs used in the primary article is archived on Zenodo.

> Dataset DOI: [https://doi.org/10.5281/zenodo.17714612](https://doi.org/10.5281/zenodo.17714612)

---

## ⚙️ Reproducibility

Each run stores:

* `parameters.json` with:

  * all simulation parameters,
  * the random seed,
  * and library versions (`numpy`, `pandas`, `networkx`, `scipy`).
* structured CSVs for agents, time series, correlations, assortativity, and network statistics.
* an aggregated index (`grid_runs_summary.csv`) that serves as a master table, linking parameter configurations to output folders.

The full grid invoked by `run_simulations.sh` matches the parameter grid documented in the Methods section of the article.

---

## 📜 License

This project is released under the **MIT License** (see `LICENSE`).
The license allows reuse and modification of the code, but any use in scientific, academic, or technical work must be accompanied by proper citation of the primary article and, when appropriate, the software and dataset records listed above.

---

## 🙏 Acknowledgements

This work is part of project PID2020-113037RB-I00, funded by MICIU/AEI/10.13039/501100011033.
Additional support from the Gobierno de Aragón (COSMOS research group, ref. T64_23R).
