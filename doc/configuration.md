# Configuration File

Configuration files manage hyperparameters of trainings.
It must provide at least the following information:
* batchsize,
* epoch,
* [dataset](#dataset) (both train / test),
* [model](#model) (network structure),
* optimizer (optimization algorithm like Adam),
* [mode](#mode) (list of configuration mode).

Additionally, we can provide the following information:
* [hook](#hook) (called before calculation by optimization algorithms at every iteration),
* [extension](#extension) (called by Trainer object of Chainer).

# Structure of config/ directory

Sample configuration files can be found under config/ directory.
```text
|- attack/
|  - configuration files for attack methods
|- outer_polytope/
|  - experiment 6.1
|- parseval_svhn/
|  - experiment 6.2
|- svhn_avg/
|  - experiment 6.2 (average-pooling ver.)
```

# Configuration of attack methods

Configuration files for attack methods are different from those for training.
They must provide the following information:
* attack method,
* args of the attack method,
* kwargs of the attack method.

They can be found under config/attack/ directory.

<a id="dataset"></a>
## Dataset

Datasets are provided under src/dataset/ directory.

Note: in this public repository, they return a pair of training and test data,
so please be careful if you start a new project using this repository.

<a id="model"></a>
## Model

Models are provided under src/model/ directory.

<a id="mode"></a>
## Mode

Mode is a list of configurations.
We provide the folloing modes:
* default (usual training)
* lmt (train using lmt (Prop.1 ver.))
* lmt_fc (train using lmt (Prop.2 ver.) activated only when lmt is also specified)
* parseval (Parseval networks)

<a id="hook"></a>
## Hook

Hooks are provided under src/hook/ directory.

<a id="extension"></a>
## Extension

Extensions are provided under src/extension/ directory.
