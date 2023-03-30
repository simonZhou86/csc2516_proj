# csc2516_proj

TODO List

- [x] Preprocessing
- [x] Loss function: Dice loss implementation
- [x] Transformer model, decoder
- [x] Attention mechanism revisit and revise
- [x] training and test scripts
- [x] baseline model(s)

After run some experiments:

- [ ] change current hard-code preceptual loss implementation to LPIPS?

## Usage

Train model (Local test)

change this line in dataset.py to reduce the time for local test

```python
Train: self.imgs/masks = self.imgs[:self.imgs.shape[0] * 1 // 1000]
Val: self.imgs/masks = self.imgs[self.imgs.shape[0] * 1 // 1000:self.imgs.shape[0] * 2 // 1000]
```


python main.py --train --dev --viz_wandb (our team name) (-- cross_att)
