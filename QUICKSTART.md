# Chest X-ray Tokenizer - Step-by-Step Quickstart Guide

This guide will walk you through using the Chest X-ray Tokenizer system from scratch.

---

## Step 1: Install Dependencies

First, install all required Python packages:

```bash
# Install dependencies
pip install -r requirements.txt
```

**Note:** This will download PyTorch and other ML libraries (~2-3 GB). The first time you run the tokenizer, it will also download a pre-trained model (~30 MB for DenseNet121).

---

## Step 2: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; import faiss; import timm; print('✓ All dependencies installed successfully')"
```

You should see: `✓ All dependencies installed successfully`

---

## Step 3: Initialize the System

Create a sample reference database to test the system:

```bash
python cli.py init --samples 20
```

**What this does:**
- Creates 20 sample embeddings with random pathologies
- Saves them to the vector database
- Creates the necessary directories (data/, embeddings/, models/, .cache/)

You should see:
```
Initializing ChestXRayTokenizer...
Creating sample database with 20 entries...
✓ Sample database created with 20 entries
```

---

## Step 4: Check Database Statistics

Verify the database was created:

```bash
python cli.py stats
```

**Expected output:**
```
============================================================
DATABASE STATISTICS
============================================================

Total embeddings: 20
Unique patients: 20
Embedding dimension: 1024

Pathology Distribution:
  • Pneumonia: 3
  • Atelectasis: 2
  • Cardiomegaly: 4
  ...
```

---

## Step 5: Using With Real Images

### Option A: Use Sample Chest X-rays (Recommended for testing)

Download a sample chest X-ray for testing:

```bash
# Create a sample data directory
mkdir -p sample_images

# Download a sample chest X-ray (example using wget or curl)
# You can use any chest X-ray image you have
```

For testing, you can use:
- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) (free, requires registration)
- [Sample chest X-rays from GitHub](https://github.com/ieee8023/covid-chestxray-dataset)
- Any PNG/JPG chest X-ray image you have

### Option B: Create a Test Image

If you just want to test the system, create a dummy grayscale image:

```bash
python quickstart_test.py
```

(We'll create this script in the next step)

---

## Step 6: Add a Reference Image

Once you have a chest X-ray image, add it to the database:

```bash
python cli.py add sample_images/chest_xray_001.png \
    --pathologies "Pneumonia,Effusion" \
    --patient-id "P12345" \
    --notes "Patient with cough and fever"
```

**Expected output:**
```
Initializing ChestXRayTokenizer...
Encoding reference image: sample_images/chest_xray_001.png
Added 1 embeddings to the vector store
✓ Added with ID: 20
```

---

## Step 7: Infer Pathologies from a New Image

Now analyze a new chest X-ray:

```bash
python cli.py infer sample_images/chest_xray_002.png \
    --k 5 \
    --show-neighbors
```

**What this does:**
- Encodes the query image
- Finds the 5 most similar images in the database
- Infers pathologies based on similar images
- Shows detailed information about similar cases

**Expected output:**
```
============================================================
INFERENCE RESULTS
============================================================

Detected Pathologies:
  • Pneumonia: 67.50%
  • Effusion: 45.20%
  • Atelectasis: 23.10%

Statistics:
  • Neighbors found: 5
  • Average similarity: 82.45%

============================================================
SIMILAR IMAGES
============================================================

1. sample_data/xray_007.png
   Similarity: 89.23%
   Pathologies: Pneumonia
   Patient: P00007

2. sample_data/xray_003.png
   Similarity: 85.67%
   Pathologies: Pneumonia, Effusion
   Patient: P00003
...
```

---

## Step 8: Using the Python API

Create a Python script to use the tokenizer programmatically:

```python
# my_script.py
from inference import PathologyInferenceEngine

# Initialize the engine
engine = PathologyInferenceEngine()

# Add a reference image
engine.add_reference_image(
    image_path="sample_images/chest_xray_001.png",
    pathologies=["Pneumonia", "Effusion"],
    patient_id="P12345"
)

# Infer pathologies from a new image
result = engine.infer_pathologies("sample_images/chest_xray_002.png")

print("Detected pathologies:", result['pathologies'])
print("Confidence scores:", result['confidence_scores'])
print(f"Found {result['num_neighbors_found']} similar cases")

# Save the database
engine.save_state()
```

Run it:
```bash
python my_script.py
```

---

## Step 9: Using the REST API

Start the API server:

```bash
python api.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
Initializing Chest X-ray Tokenizer System...
✓ System initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test the API

Open a new terminal and try these commands:

**1. Check API health:**
```bash
curl http://localhost:8000/health
```

**2. Infer pathologies:**
```bash
curl -X POST "http://localhost:8000/infer" \
  -F "file=@sample_images/chest_xray.png" \
  -F "return_neighbors=true"
```

**3. Add reference image:**
```bash
curl -X POST "http://localhost:8000/add_reference" \
  -F "file=@sample_images/chest_xray.png" \
  -F "pathologies=Pneumonia,Effusion" \
  -F "patient_id=P12345"
```

**4. Get statistics:**
```bash
curl http://localhost:8000/statistics
```

**5. View API docs:**
Open your browser to: http://localhost:8000/docs

---

## Step 10: Building a Production Database

To use this system in production, follow these steps:

### 1. Collect labeled chest X-rays
```bash
# Organize your images
mkdir -p data/chest_xrays
# Copy your labeled images to this directory
```

### 2. Create a CSV with labels
```csv
image_path,pathologies,patient_id,notes
data/chest_xrays/img001.png,"Pneumonia",P001,"Initial presentation"
data/chest_xrays/img002.png,"Atelectasis,Effusion",P002,"Follow-up"
data/chest_xrays/img003.png,"No Finding",P003,"Routine screening"
```

### 3. Bulk import (create a script)
```python
import pandas as pd
from inference import PathologyInferenceEngine

# Load labels
df = pd.read_csv('labels.csv')

# Initialize engine
engine = PathologyInferenceEngine()

# Add all images
for _, row in df.iterrows():
    pathologies = row['pathologies'].split(',')
    engine.add_reference_image(
        image_path=row['image_path'],
        pathologies=pathologies,
        patient_id=row['patient_id'],
        notes=row.get('notes', '')
    )
    print(f"✓ Added {row['image_path']}")

# Save
engine.save_state()
print(f"✓ Added {len(df)} images to database")
```

---

## Common Issues & Solutions

### Issue 1: "No module named 'torch'"
**Solution:** Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### Issue 2: "CUDA out of memory"
**Solution:** Use CPU mode by editing `config.py`:
```python
# Add at the top of config.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

### Issue 3: "No similar images found"
**Solution:** Your database might be empty or threshold is too high. Check:
```bash
python cli.py stats  # Verify database has images
```

Or lower the similarity threshold in `config.py`:
```python
SIMILARITY_THRESHOLD = 0.3  # Lower threshold
```

### Issue 4: Model download is slow
**Solution:** The first run downloads a ~30MB model. This is one-time. If it fails:
```python
# Manually download in Python:
import timm
model = timm.create_model('densenet121', pretrained=True)
```

---

## Next Steps

### For Better Accuracy:

1. **Use medical imaging pre-trained models:**
   - Fine-tune on ChestX-ray14 or CheXpert datasets
   - Use models pre-trained on medical images

2. **Build a larger reference database:**
   - More reference images = better inference
   - Aim for at least 1000+ labeled images per pathology

3. **Adjust parameters:**
   - Increase k_neighbors for more consensus
   - Adjust similarity_threshold based on your use case

### For Production Use:

1. **Add authentication to the API**
2. **Set up proper logging and monitoring**
3. **Use a production WSGI server (gunicorn)**
4. **Deploy with Docker for easy scaling**
5. **Add data validation and error handling**

---

## What Each Command Does

| Command | What It Does | When To Use |
|---------|-------------|-------------|
| `cli.py init` | Create sample database | First time setup, testing |
| `cli.py add` | Add labeled X-ray | Building reference database |
| `cli.py infer` | Analyze new X-ray | Making predictions |
| `cli.py stats` | Show database info | Check database status |
| `cli.py encode` | Get embedding vector | Advanced use, debugging |
| `cli.py clear` | Delete all data | Start fresh |
| `api.py` | Start web server | Web/mobile integration |
| `example_usage.py` | Run all examples | Learning the system |

---

## Getting Help

- **Check examples:** `python example_usage.py`
- **API documentation:** http://localhost:8000/docs (when API is running)
- **Read the code:** All modules have detailed docstrings
- **Test with sample data:** Use `cli.py init` to create test data

---

## Summary

**Minimum steps to get started:**
```bash
# 1. Install
pip install -r requirements.txt

# 2. Initialize
python cli.py init --samples 20

# 3. Test
python cli.py stats

# 4. Done! Now you can add real images and infer pathologies
```

**Most common workflow:**
```bash
# Add reference images (do this many times)
python cli.py add my_xray.png --pathologies "Pneumonia"

# Analyze new images
python cli.py infer unknown_xray.png --show-neighbors
```

That's it! You now have a working chest X-ray analysis system. 🎉
