# OpenDataArena-Tool

<p align="center">
  <img src="docs/imgs/OpenDataLab.png" width="300px" style="vertical-align:middle;">
  <br />
  <br />
  <a href="https://github.com/OpenDataArena/OpenDataArena-Tool"><img alt="stars" src="https://img.shields.io/github/stars/OpenDataArena/OpenDataArena-Tool" /></a>
  <a href="https://github.com/OpenDataArena/OpenDataArena-Tool"><img alt="forks" src="https://img.shields.io/github/forks/OpenDataArena/OpenDataArena-Tool" /></a>
  <a href="https://github.com/OpenDataArena/OpenDataArena-Tool/issues"><img alt="open issues" src="https://img.shields.io/github/issues-raw/OpenDataArena/OpenDataArena-Tool" /></a>
  <a href="https://github.com/OpenDataArena/OpenDataArena-Tool/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <!-- <a href="https://github.com/OpenDataArena/OpenDataArena-Tool/releases">
    <img alt="Latest Release" src="https://img.shields.io/github/release/OpenDataArena/OpenDataArena-Tool.svg" />
  </a> -->
  <a href="https://opendataarena-tool.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/opendataarena-tool/badge/?version=latest" /></a>
</p>


## What's New
- 2025-07-26: We release the [OpenDataArena](https://opendataarena.github.io/) platform and the [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) repository.

## Overview
[OpenDataArena (ODA)](https://opendataarena.github.io/) is an open, transparent, and extensible platform for evaluating the value of post-training datasets, aiming to make every dataset measurable, comparable, and verifiable.

This repository includes the tools for ODA platform:
- [Data Scoring](./data_scorer): Assess datasets through diverse metrics and methods, including model-based methods, llm-as-judge, and heuristic methods.
- [Model Training](./model_train): Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to supervised fine-tuning (SFT) the model on the datasets. We provide the SFT scripts for reproducible experiments on mainstream models and benchmarks.
- [Benchmark Evaluation](./model_eval): Use [OpenCompass](https://github.com/open-compass/opencompass) to evaluate the performance of the model on popular benchmarks from multiple domains (math, code, science, and general instruction). We also provide the evaluation scripts for the datasets in ODA.

## Quick Start
First, clone the repository and its submodules:
```bash
git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git --recursive
cd OpenDataArena-Tool
```
Then, you can start to use the tools in ODA:
* To score your own dataset, please refer to [Data Scoring](./data_scorer) for more details.
* To train the models on the datasets in ODA, please refer to [Model Training](./model_train) for more details.
* To evaluate the models on the benchmarks in ODA, please refer to [Benchmark Evaluation](./model_eval) for more details.
 
## Acknowledgments
We thank the following projects for their contributions to ODA:
* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* [OpenCompass](https://github.com/open-compass/opencompass)
* [vllm](https://github.com/vllm-project/vllm)
* [evalplus](https://github.com/evalplus/evalplus)
* [xVerify](https://github.com/IAAR-Shanghai/xVerify)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Citation
If you find this project useful, please consider citing:

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```

<!-- ## Star History
![Star History Chart](https://api.star-history.com/svg?repos=OpenDataArena/OpenDataArena-Tool&type=Date) -->
