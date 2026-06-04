```
       _____
______ ___(_)________      _______ _____  __
_  __ `/_  /_  __ \_ | /| / /  __ `/_  / / /
/ /_/ /_  / / /_/ /_ |/ |/ // /_/ /_  /_/ /
\__,_/ /_/  \____/____/|__/ \__,_/ _\__, /
                                   /____/
```

# 🛣️ AioWay

> Aioway is an optimizing compiler for deep learning algorithms.
> It treats the machine learning models / algorithms as instructions and build the pipeline that way.

### 🏢 Architecture

### 🍰 Features

> Most of these features are done but not polished yet! But will be in couple of weeks.

- Fast. Compared to neural architecture search, the optimization can be rule based (fast).
- Detects the tasks at hand, resource available, and select the best algorithms and models.
- The models built from aioway would be white box (explainable), due to our architecture.
- Allows upgrading parts of the models. You scale up to different model size, and to different machines.
- Both relational algebra and python library interface.
- Extensible with custom pytorch.


### ⛰️ Compared with current landscape

1. Neural architecture search: Too slow (because most need backtracking).
2. Current autoML framework: non flexible enough, usually stuck with UI or fixed set of models (black box still).
3. Pretrained models / LLM: Usually expensive, non explainable, less flexible.

## 🌟 Give us a star!

That's all for now!

If you have read this far, please consider giving me a star (⭐) or a fork (🍴).

This will keep my motivation going!

Or if you have too much cash at hand: [![BuyMeACoffee](https://raw.githubusercontent.com/pachadotdev/buymeacoffee-badges/main/bmc-yellow.svg)](https://www.buymeacoffee.com/rentruewang)

### 🗺️ Roadmap

We are most likely launching `v0.1.0` before July 2026, but before that,
see the pre-release tracking [project](https://github.com/users/rentruewang/projects/7) for more details.



## 🤔 Why aioway yada yada

In the recent years, machine learning's entry barrier has gotten higher, rather than lower. With the increasing number of algorithms and libraries and models, it's no wonder qualified data scientists are rare because you would need years of training to keep up to the status quo.

We designed Aioway in a way such that instead of thinking about **how** to do ML, you specify **what** to do. Instead of focusing on what algorithms and models to use, Aioway allows you to focus on the use cases by taking into account the context of the problem, and perform compliation according to the data to ensure good performance. Automatically.

## 🤝 Contributing

Contributing is of course welcome. Please see the [contributing guide](./CONTRIBUTING.md) and follow the [code of conduct](./CODE_OF_CONDUCT.md).

## 👨‍👨‍👦‍👦 Contributors

<a href="https://github.com/rentruewang/aioway/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=rentruewang/aioway" />
</a>



## 🐨 Relation to koila

Aioway builds on top of the original [`koila` (moved to a branch)][koila]. The `torch` team built `FakeTensor` which overlaps a lot with `koila`'s functionality, so it's no longer maintained. See the rationale in the [koila][koila] branch.

Conceptually, `aioway` works in a similar way, but instead of `Tensor` ops, `aioway` focuses on a higher level, on **algorithm building**.


[koila]: https://github.com/rentruewang/aioway/tree/koila
