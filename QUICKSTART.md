# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Add Model Checkpoints

Place your trained model files in the `checkpoints/` directory:
- `scnn_best.pth`
- `polylane_best.pth`
- `ultrafast_best.pth`

### Step 3: Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Step 4: Test with Sample Image

1. Click "Browse files" in the sidebar
2. Upload a road image
3. Click "🚀 Detect Lanes"
4. View results from all three models!

## 🎯 CLI Usage

```bash
# Process single image
python inference.py --image test.jpg

# Process video
python inference.py --video test.mp4

# Batch process directory
python inference.py --input-dir images/ --output-dir results/
```

## 📝 Next Steps

- Read `README.md` for detailed documentation
- Check `docs/API.md` for API reference
- See `docs/TRAINING.md` for training your own models

## ❓ Troubleshooting

**Models not loading?**
- Ensure checkpoints are in `checkpoints/` directory
- Check that .pth files contain `model_state_dict` key

**Out of memory?**
- Reduce batch size in video processing
- Use CPU instead of GPU for testing

**Import errors?**
- Verify all dependencies installed: `pip list`
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## 💡 Tips

- Use GPU for faster processing
- Start with images before videos
- Adjust confidence threshold for better results
- Try different line thickness values for visualization

---

For detailed documentation, see [README.md](README.md)
