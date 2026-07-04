## Codebase Structure

As described above, AutoSpeed is a plug-and-play training scheme that can be applied to various non-generative and generative embodied manipulation policies. To make it easier to use, we have reorganized the development code and refactored AutoSpeed into an **independent server class**, integrating its core functionalities into `./agent/autospeed_server.py`. This allows you to quickly deploy AutoSpeed in the training pipelines of different policies.

The `example` folder provides policy examples that we have already adapted. This folder will be gradually updated with more examples in the future.

## To be released

We will gradually reorganize and release the already adapted models. Of course, the current version already includes all the necessary modules. You only need to add a selective optimization process before gradient optimization in the models that need to be adapted.