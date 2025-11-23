# Training Guide

This guide explains how to train the lane detection models from scratch.

## Dataset Preparation

### TuSimple Dataset

1. **Download the dataset**
```bash
# Visit: https://github.com/TuSimple/tusimple-benchmark
```

2. **Dataset structure**
```
data/tusimple/
├── clips/
│   ├── 0313-1/
│   ├── 0313-2/
│   └── ...
├── label_data_0313.json
├── label_data_0601.json
└── test_label.json
```

## Training Configuration

### SCNN

```yaml
model: SCNN
epochs: 50
batch_size: 16
learning_rate: 0.001
optimizer: Adam
loss: WeightedCrossEntropy
```

### PolyLaneNet

```yaml
model: PolyLaneNet
epochs: 50
batch_size: 16
learning_rate: 0.0001
optimizer: Adam
loss: MSE + BCE
```

### UltraFast

```yaml
model: UltraFast
epochs: 50
batch_size: 32
learning_rate: 0.0005
optimizer: SGD
```

## Training Command

```bash
# Example training command (requires implementing train.py)
python train.py \
    --model scnn \
    --data-dir data/tusimple \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --save-dir checkpoints/
```

## Tips for Better Training

1. **Start with pretrained backbone**: Use ImageNet pretrained weights
2. **Learning rate scheduling**: Use cosine annealing
3. **Data augmentation**: Essential for generalization
4. **Mixed precision**: Faster training with minimal accuracy loss
5. **Early stopping**: Monitor validation loss

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

### Poor Convergence
- Adjust learning rate
- Verify data preprocessing
- Check loss function implementation

### Overfitting
- Add more augmentation
- Increase dropout
- Use weight decay
