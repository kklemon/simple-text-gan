simple-text-gan
===============

Implementation of a simple byte-level text GAN using WGAN-GP loss.
No tokenization, no autoregressive sampling, no tricks such as gradient estimation or gumbel softmax.
Just predicting plain bytes.

Usage
-----

```shell
python main.py \
  --config=configs/config.yaml \
  fit \
  --data.path=data/ptb.train.txt \
  --trainer.gpus=1
```
Samples
-------

```
some reachnry on wis set 's mum
<unk> suprentals the last N N o
under other conseres ms. hell r
mr. mitcentenn shest devilas al
all alling votings recently low
given employks resultsscal of e
on mr. posoctib al reeterg
paring finiicounters this rate
the computer 's also willical n
in <unk> triak was the since wa
tr. prrc blation of a states o
the minn'aicomplecely ho said i
bnded you N bonds do n't <unk>
on <unk> with the find look the
for the week by houstls and pep
but do n't wern wockers on enti

```

Results are mediocre at best so far but this repository should serve as a base for future experiments.

Requirements
------------

* PyTorch
* PyTorch Lightning

References
----------

```bibtex
@misc{gulrajani2017improved,
    title={Improved Training of Wasserstein GANs},
    author={Ishaan Gulrajani and Faruk Ahmed and Martin Arjovsky and Vincent Dumoulin and Aaron Courville},
    year={2017},
    eprint={1704.00028},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```