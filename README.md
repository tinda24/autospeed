
The corresponding experiment videos are available at: [Project Page](https://zihengqiu.github.io/AutoSpeed/)
### News

[26-06-20] "AutoSpeed: Annotation-Free Stage-Adaptive Motion Speed Learning for Robot Manipulation" has been accepted by European Conference on Computer Vision **(ECCV) 2026**.

[26-07-02] We have open-sourced the core code required for AutoSpeed training, so you can try integrating it into your own codebase. Recommend: [ACT](https://github.com/tonyzhaozh/act) [BAKU](https://github.com/siddhanthaldar/BAKU/tree/main)


## Codebase Structure

**AutoSpeed** is a model-agnostic learning framework that enables existing visuomotor policies to predict trajectories with stage-adaptive motion speeds, without requiring speed or stage annotations. 

<p align="center">
  <video src="./assets/real-world-experiment.mp4" controls width="80%">
    real-world-experiment.
  </video>
</p>

As described above, AutoSpeed is a plug-and-play training scheme that can be applied to various non-generative and generative embodied manipulation policies. To make it easier to use, we have reorganized the development code and refactored AutoSpeed into an **independent server class**, integrating its core functionalities into `./agent/autospeed_server.py`. This allows you to quickly deploy AutoSpeed in the training pipelines of different policies.

The `example` folder provides policy examples that we have already adapted. This folder will be gradually updated with more examples in the future.

## Quick Start

We will gradually reorganize and release the already adapted models. Of course, the current version already includes all the necessary modules. You only need to add a selective optimization process before gradient optimization in the models that need to be adapted.

## Checkpoints

To be released.