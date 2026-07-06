
The corresponding experiment videos are available at: [Project Page](https://zihengqiu.github.io/AutoSpeed/)
### News

[26-06-20] "AutoSpeed: Annotation-Free Stage-Adaptive Motion Speed Learning for Robot Manipulation" has been accepted by European Conference on Computer Vision **(ECCV) 2026**.

[26-07-02] We have open-sourced the core code required for AutoSpeed training, so you can try integrating it into your own codebase. Recommend: [ACT](https://github.com/tonyzhaozh/act) [BAKU](https://github.com/siddhanthaldar/BAKU/tree/main)

[26-07-05] We have open-sourced the core code and pretrained checkpoints in our real-world experiments. See in [[Page]](./example/README.md)


## Codebase Structure

**AutoSpeed** is a model-agnostic learning framework that enables existing visuomotor policies to predict trajectories with stage-adaptive motion speeds, without requiring speed or stage annotations. 

As described above, AutoSpeed is a plug-and-play training scheme that can be applied to various non-generative and generative embodied manipulation policies. To make it easier to use, we have reorganized the development code and refactored AutoSpeed into an **independent server class**, integrating its core functionalities into `./agent/autospeed_server.py`. This allows you to quickly deploy AutoSpeed in the training pipelines of different policies.

The `example` folder provides policy examples that we have already adapted. This folder will be gradually updated with more examples in the future.

## Real-World Experiment

<p align="center">
  <a href="https://zihengqiu.github.io/AutoSpeed/">
    <img src="./assets/real-world-experiment.gif" width="80%" alt="Real-World Experiment">
  </a>
</p>

<p align="center">
  <a href="https://zihengqiu.github.io/AutoSpeed/">
    ▶ Watch full experiment videos
  </a>
</p>

## Quick Start

We will gradually reorganize and release the already adapted models. See the process in the news.


## Acknowledgements and Code References

This repository builds upon and refers to several excellent open-source projects. We sincerely thank the authors of the following repositories for releasing their code and contributing to the robotics and decision-making research community:

* [ACT](https://github.com/tonyzhaozh/act): simulation baseline.
* [BAKU](https://github.com/siddhanthaldar/BAKU): parts of our policy implementation, training pipeline, and experimental code structure are adapted from or inspired by this repository.
* [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser): parts of our diffusion-related implementation and training utilities are adapted from or inspired by this modular diffusion-model codebase.
* [COCOS](https://github.com/ZibinDong/cocos): parts of our diffusion-related implementation and training utilities are adapted from or inspired by this modular diffusion-model codebase.
* [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): our simulated manipulation experiments and evaluation protocols are built on or reference this lifelong robot learning benchmark.
* [MetaWorld](https://github.com/Farama-Foundation/Metaworld): our multi-task manipulation experiments and evaluation environments are built on or reference this benchmark suite.