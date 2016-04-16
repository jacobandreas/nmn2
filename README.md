# Neural module networks: SHAPES experiments


This branch contains data and horrible, slow Theano code for the SHAPES
experiments described in the paper:

> [Neural module networks](http://arxiv.org/abs/1511.02799). Jacob Andreas,
> Marcus Rohrbach, Trevor Darrell and Dan Klein. CVPR 2016. 

For details about the general NMN framework (and code for other tasks), look at
the `master` branch.

To run our experiments, call

    python main.py

You will need [Lasagne v0.1](https://pypi.python.org/pypi/Lasagne/0.1) and its
dependencies.

The optimizer seems to occasionally get stuck in local optima, so try restarting
if it doesn't converge the first time.

Code used to generate the dataset can be found in the `data` directory. See
`visualize.py` for examples of how to array data into actual images.
