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
  <br />
  <br />
  <a href="README.md">English</a> | 简体中文
</p>


## 最新动态
- 2025-07-26: 我们发布了 [OpenDataArena](https://opendataarena.github.io/) 平台和 [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) 仓库。

## 概览
[OpenDataArena (ODA)](https://opendataarena.github.io/) 是一个开源、透明、可扩展的平台，用于评估后训练数据集的价值，旨在使每个数据集可衡量、可比较和可验证。

这个仓库包括了 ODA 平台的工具：
- [Data Scoring](./data_scorer): 通过多种指标和方法评估数据集，包括基于模型的方法、llm-as-judge 和启发式方法。
- [Model Training](./model_train): 使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 在数据集上进行监督微调 (SFT)。我们提供了 SFT 脚本，用于在主流模型和基准上进行可重复实验。
- [Benchmark Evaluation](./model_eval): 使用 [OpenCompass](https://github.com/open-compass/opencompass) 评估模型在多个领域（数学、代码、科学和通用）的流行基准上的性能。我们还提供了 ODA 中数据集的评估脚本。

## 快速开始
首先，克隆仓库及其子模块：
```bash
git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git --recursive
cd OpenDataArena-Tool
```
然后，您可以开始使用 ODA 中的工具：
* 要评估您自己的数据集，请参阅 [Data Scoring](./data_scorer) 了解更多详细信息。
* 要在 ODA 中的数据集上训练模型，请参阅 [Model Training](./model_train) 了解更多详细信息。
* 要在 ODA 中的基准上评估模型，请参阅 [Benchmark Evaluation](./model_eval) 了解更多详细信息。

## Contributors
我们感谢这些杰出的研究人员和开发人员对 OpenDataArena 项目的贡献。欢迎合作和贡献！
<p align="center">
  <a href="https://github.com/gavinwxy" title="Xiaoyang Wang"><img src="docs/avatars_circle/gavinwxy.png" width="60" alt="Xiaoyang Wang" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/QizhiPei" title="Qizhi Pei"><img src="docs/avatars_circle/QizhiPei.png" width="60" alt="Qizhi Pei" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/orangeadegit" title="Mengzhang Cai"><img src="docs/avatars_circle/orangeadegit.png" width="60" alt="Mengzhang Cai" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/Word2VecT" title="Zinan Tang"><img src="docs/avatars_circle/Word2VecT.png" width="60" alt="Zinan Tang" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/Leey21" title="Yu Li"><img src="docs/avatars_circle/Leey21.png" width="60" alt="Yu Li" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/Bl404ue" title="Mengyuan Sun"><img src="docs/avatars_circle/Bl404ue.png" width="60" alt="Mengyuan Sun" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/LHL3341" title="Honglin Lin"><img src="docs/avatars_circle/LHL3341.png" width="60" alt="Honglin Lin" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/GX-XinGao" title="Xin Gao"><img src="docs/avatars_circle/GX-XinGao.png" width="60" alt="Xin Gao" style="border-radius: 50%; margin: 4px;"></a>
  <br />
  <br />
  <a href="https://github.com/apeterswu" title="Lijun Wu"><img src="docs/avatars_circle/apeterswu.png" width="60" alt="Lijun Wu" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/pzs19" title="Zhuoshi Pan"><img src="docs/avatars_circle/pzs19.png" width="60" alt="Zhuoshi Pan" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/ming-bot" title="Chenlin Ming"><img src="docs/avatars_circle/ming-bot.png" width="60" alt="Chenlin Ming" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/ChampionZhong" title="Zhanping Zhong"><img src="docs/avatars_circle/ChampionZhong.png" width="60" alt="Zhanping Zhong" style="border-radius: 50%; margin: 4px;"></a>
  <a href="https://github.com/conghui" title="Conghui He"><img src="docs/avatars_circle/conghui.png" width="60" alt="Conghui He" style="border-radius: 50%; margin: 4px;"></a>
</p>



## 许可证
本项目采用 MIT 许可证 - 请参阅 [LICENSE](./LICENSE) 文件了解更多详细信息。

## 引用
如果您觉得这个项目有用，请考虑引用：

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
