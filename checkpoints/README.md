# Model Checkpoints

Place your trained model weights here:

## Required Files

- `scnn_best.pth` - SCNN model weights
- `polylane_best.pth` - PolyLaneNet model weights
- `ultrafast_best.pth` - UltraFast model weights

## Checkpoint Format

Each checkpoint should contain a dictionary with:

```python
{
    'model_state_dict': ...,      # Model parameters
    'optimizer_state_dict': ...,  # Optimizer state (optional)
    'epoch': ...,                 # Training epoch
    'loss': ...,                  # Best validation loss
}
```

## Training Your Own Models

See `docs/TRAINING.md` for instructions on training your own models.

## Download Pretrained Models

If you have pretrained models hosted somewhere, add the download link here:

```bash
# Example download commands
wget https://your-hosting-url/scnn_best.pth
wget https://your-hosting-url/polylane_best.pth
wget https://your-hosting-url/ultrafast_best.pth
```
