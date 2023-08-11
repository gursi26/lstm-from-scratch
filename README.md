# lstm-from-scratch
A from scratch PyTorch implementation Multilayered LSTM for character level language modelling

[Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) </br>
[Dataset](http://www.manythings.org/bilingual/)

Sample outputs after brief training:
```python
>>> print(sample(model, dataset, "thou", 512, 1000, dev, 3))
"""thoues time's toy chelding
  if fare whith eress cond ie ment feem,
  when ell the piere that thou art bedouted mead then times
    so thou, thy self out-going in thy noon
    unlooked thee of the stick sweet nead dother, be the thee,
  and buanteor sail math loft thou art moch toof,
  and yet methanks i have astronomy,
  but not to tell of good, or eve thy brood,"""
```
