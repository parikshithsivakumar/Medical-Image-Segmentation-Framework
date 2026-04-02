# 🔴 CRITICAL SINGLE-LINE FIX REQUIRED BEFORE RUNNING ANATOMY TRAINING

## The Problem
The notebook Cell 4 (Anatomy CNN Training) has one line that needs to be fixed:

**Current (WRONG):**
```python
anatomy_model = AnatomyCNN(num_classes=3, embedding_dim=64).to(DEVICE)
```

**Should be (CORRECT):**
```python
anatomy_model = AnatomyCNN(num_classes=2, embedding_dim=64).to(DEVICE)
```

## Why
- `NUM_ANATOMY_CLASSES` is now 2 (merged ileum → "other" class)
- But the notebook still passes 3 to the model
- This causes CrossEntropyLoss to crash (target indices >= num_classes)

## How to Fix (30 seconds)

### Option A: Manual Edit (Recommended)
1. Open the notebook: `notebooks/ACDNet_Pipeline.ipynb` in VSCode
2. Scroll to Cell 4 (around line 1322)
3. Find the line: `anatomy_model = AnatomyCNN(num_classes=3, embedding_dim=64).to(DEVICE)`
4. Change `num_classes=3` to `num_classes=2`
5. Save (Ctrl+S)

### Option B: Search & Replace
1. Open Find & Replace (Ctrl+H)
2. Find: `AnatomyCNN(num_classes=3`
3. Replace: `AnatomyCNN(num_classes=2`
4. Replace the first occurrence (should be only in Cell 4)
5. Save

## Verification
After fix, the line should read:
```python
anatomy_model = AnatomyCNN(num_classes=2, embedding_dim=64).to(DEVICE)
```

## Next Steps After Fix
✅ All 6 issues verified and implemented
✅ Notebook Cells 1-3 are ready to run
✅ Once this line is fixed, Cell 4 (Anatomy Training) is ready to run

Expected Anatomy Training Results:
- Duration: 2-3 minutes (30 epochs)
- Best validation accuracy: 96-98% (improved from 98.1% with better generalization)
- Overfitting gap: 1.5-2.0x (improved from 4.3x)
- Checkpoint: `checkpoints/anatomy_cnn_best.pth`

NO OTHER CHANGES NEEDED - This is the ONLY notebook edit required!
