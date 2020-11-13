## DynIm (Dynamic-Importance Sampling), Version 0.1

##### Authors: Harsh Bhatia (hbhatia@llnl.gov) and Joseph Y Moon
##### Released: Nov 12, 2020

`dynim` is a pure-python package to perform ***dynamic-importance (DynIm)
sampling*** on a high-dimensional data set.

DynIm is designed to minimize redundancy and maximize the coverage of the
sampled points. DynIm uses the notion of "*dissimilarity*" from previously
selected samples to define the importance of potential selections, and selects
the ones that are most dissimilar. Simply, DynIm provides a *farthest-point*
sampling approach.

Currently, `dynim` uses L2 distances in the given high-dimensional space to
define similarity and can be configured to use *exact* as well as *approximate*
distances. Approximate distances are useful for computational viability for
large data sizes and large data dimensionality. `dynim` also provides a random
sampler for comparison of sampling quality.


#### Dependencies

`dynim` uses [`faiss`](https://github.com/facebookresearch/faiss) to implement
nearest neighbor searches for sampling, and has been tested with `faiss v1.6.3`.
Currently, we ask the user to install `faiss` explicitly from source. Please
see [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)
for installation instructions..

Other dependencies are `numpy` and `pyyaml` (if needed, will be installed with
`dynim`).

#### Installation

Once the dependencies are installed, `dynim` can be installed as follows:

```
git clone git@github.com:LLNL/dynim.git
cd dynim
pip install .
```

Please test your installation as follows.
```
python3 -m unittest examples/test_dynim.py
```

### Examples

See the `examples` directory.

### License

dynim is distributed under the terms of the MIT license.

See [LICENSE](./LICENSE) and [NOTICE](./NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE-813147
