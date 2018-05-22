# LMT
Public code for a paper ["Lipschitz-Margin Training:
Scalable Certification of Perturbation Invariance
for Deep Neural Networks."](https://arxiv.org/abs/1802.04034)

### update:
Pre-print and code were updated to version 2 (2018/5/23).

How to train:
```commandline
python3 train.py (configuration-file-name).py
```
[what is the configuration file?](doc/configuration.md)

example:
```commandline
python3 train.py config/parseval_svhn/default.py --gpu 0
```

How to evaluate with attacks:
```commandline
python3 evaluate.py (result-dir-of-trained-network) (attack-configuration).py
```
This script calculates each value in inequality (5) in Sec.6
and create inequality-(attack_name).npy under a specified result directory.

[what is the result directory?](doc/result_dir.md)

example:
```commandline
python3 evaluate.py result/config/parseval_svhn/default-00 config/attack/cw_100_1.py
```

# Reference
Y. Tsuzuku, I. Sato, M. Sugiyama:
Lipschitz-Margin Training:
Scalable Certification of Perturbation Invariance for Deep Neural Networks,
(2018), [url](https://arxiv.org/abs/1802.04034), [bibtex](lmt_bibtex.txt)
