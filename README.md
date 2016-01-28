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
straightforward to implement---alternative layout predictors, supervised
training of specific modules, etc. 

If you use this code, please cite the arXiv submission above. Feel free to email
me at [jda@cs.berkeley.edu](mailto:jda@cs.berkeley.edu) if you have questions.
