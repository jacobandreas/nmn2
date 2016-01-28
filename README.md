# Neural module networks

This library provides code for training and evaluating _neural module networks_
(NMNs). An NMN is a neural network that is assembled dynamically by composing
shallow network fragments called _modules_ into a deeper structure. These
modules are jointly trained to be freely composable. For a general overview to
the framework, refer to:

> [Learning to compose neural networks for question answering](http://arxiv.org/abs/1601.01705).
> Jacob Andreas, Marcus Rohrbach, Trevor Darrell and Dan Klein.
> _arXiv:1601.01705_.

At present the code supports predicting network layouts from natural-language
strings, with end-to-end training of modules. Various extensions should be
straightforward to implement&mdash;alternative layout predictors, supervised
training of specific modules, etc. 

If you use this code, please cite the arXiv submission above. Feel free to email
me at [jda@cs.berkeley.edu](mailto:jda@cs.berkeley.edu) if you have questions.

## Installing dependencies

You will need to build **my fork** of the excellent
[ApolloCaffe](http://apollocaffe.com/) library. This fork may be found at
[jacobandreas/apollocaffe](https://github.com/jacobandreas/apollocaffe), and 
provides support for a few Caffe layers that haven't made it into the main 
Apollo repository. Ordinary Caffe users: note that you will have to install the
`runcython` Python module in addition to the usual Caffe dependencies.

One this is done, update `APOLLO_ROOT` at the top of `run.sh` to point to your
ApolloCaffe installation. 

## Downloading data

TODO.

## Running experiments

Different experiments can be run by providing an appropriate configuration file
on the command line (see the last line of `run.sh`). Various examples for VQA,
Shapes, and GeoQA are provided in the `config` directory.
