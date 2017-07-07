# Neural module networks

**UPDATE 22 Jun 2017: Code for our end-to-end module network framework is 
available at https://github.com/ronghanghu/n2nmn. The n2nmn code works better
and is easier to set up. Use it!**

This library provides code for training and evaluating _neural module networks_
(NMNs). An NMN is a neural network that is assembled dynamically by composing
shallow network fragments called _modules_ into a deeper structure. These
modules are jointly trained to be freely composable. For a general overview to
the framework, refer to:

> [Neural module networks](http://arxiv.org/abs/1511.02799).
> Jacob Andreas, Marcus Rohrbach, Trevor Darrell and Dan Klein.
> CVPR 2016.

<!-- -->
> [Learning to compose neural networks for question
> answering](http://arxiv.org/abs/1601.01705).
> Jacob Andreas, Marcus Rohrbach, Trevor Darrell and Dan Klein.
> NAACL 2016.

At present the code supports predicting network layouts from natural-language
strings, with end-to-end training of modules. Various extensions should be
straightforward to implement&mdash;alternative layout predictors, supervised
training of specific modules, etc. 

Please cite the CVPR paper for the general NMN framework, and the NAACL paper
for dynamic structure selection. Feel free to email me at
[jda@cs.berkeley.edu](mailto:jda@cs.berkeley.edu) if you have questions.  This
code is released under the Apache 2 license, provided in `LICENSE.txt`.

## Installing dependencies

You will need to build **my fork** of the excellent
[ApolloCaffe](http://apollocaffe.com/) library. This fork may be found at
[jacobandreas/apollocaffe](https://github.com/jacobandreas/apollocaffe), and 
provides support for a few Caffe layers that haven't made it into the main 
Apollo repository. Ordinary Caffe users: note that you will have to install the
`runcython` Python module in addition to the usual Caffe dependencies.

One this is done, update `APOLLO_ROOT` at the top of `run.sh` to point to your
ApolloCaffe installation.

You will also need to install the following packages:

    colorlogs, sexpdata

## Downloading data

All experiment data should be placed in the `data` directory.

#### VQA

In `data`, create a subdirectory named `vqa`. Follow the [VQA setup
instructions](https://github.com/VT-vision-lab/VQA/blob/master/README.md) to
install the data into this directory. (It should have children `Annotations`,
`Images`, etc.)

We have modified the structure of the VQA `Images` directory slightly. `Images`
should have two subdirectories, `raw` and `conv`. `raw` contains the original
VQA images, while `conv` contains the result of preprocessing these images with
a [16-layer VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) as
described in the paper. Every file in the `conv` directory should be of the form
`COCO_{SETNAME}_{IMAGEID}.jpg.npz`, and contain a 512x14x14 image map in zipped 
numpy format. Here's a [gist](https://gist.github.com/jacobandreas/897987ac03f8d4b9ea4b9e44affa00e7)
with the code I use for doing the extraction.

#### GeoQA

Download the GeoQA dataset from the [LSP
website](http://rtw.ml.cmu.edu/tacl2013_lsp/), and unpack it into `data/geo`.

## Parsing questions

Every dataset fold should contain a file of parsed questions, one per line,
formatted as S-expressions. If multiple parses are provided, they should be
semicolon-delimited. As an example, for the question "is the train modern" we
might have:

    (is modern);(is train);(is (and modern train))

For VQA, these files should be named `Questions/{train2014,val2014,...}.sps2`.
For GeoQA, they should be named `environments/{fl,ga,...}/training.sps`. Parses
used in our papers are provided in `extra` and should be installed in the
appropriate location. The VQA parser script is also located under `extra/vqa`;
instructions for running are provided in the body of the script.

## Running experiments

You will first need to create directories `vis` and `logs` (which respectively
store run logs and visualization code)

Different experiments can be run by providing an appropriate configuration file
on the command line (see the last line of `run.sh`). Examples for VQA and GeoQA
are provided in the `config` directory.

Looking for SHAPES? I haven't finished integrating it with the rest of the 
codebase, but check out the `shapes` branch of this repository for data and 
code.

## TODO

- Configurable data location
- Model checkpointing
