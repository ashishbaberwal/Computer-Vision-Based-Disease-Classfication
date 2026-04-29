## 🎯 What This Project Does

Given a chest X-ray image, this system:

1. **Classifies** the X-ray into one of 4 categories using a 3-model ensemble
2. **Retrieves** relevant medical knowledge from textbooks using Hybrid RAG
3. **Generates** a structured 5-section clinical radiology report using an LLM

```
📷 Input X-Ray
      ↓
🧠 Ensemble (ResNet50 + ResNet101 + DenseNet121)
      ↓
🔀 Hybrid RAG (ChromaDB + BM25) → Felson\'s + ESR Textbooks
      ↓
🤖 LLM (Gemini 1.5 Pro / GPT-4o / Ollama)
      ↓
📄 Structured Clinical Report
```

---

## 🏷️ Classes

| Class | Description |
|---|---|
| 🦠 **COVID-19** | Ground-glass opacity, bilateral peripheral infiltrates |
| 🫁 **Lung Opacity** | Consolidation, pleural effusion, airspace opacity |
| ✅ **Normal** | Clear lung fields, normal cardiac silhouette |
| 🤧 **Viral Pneumonia** | Interstitial pattern, peribronchial thickening |

---

## ✨ Features

- ✅ **3-Model Ensemble** — ResNet50 + ResNet101 + DenseNet121 with weighted softmax averaging
- ✅ **Smart Load-or-Train** — loads saved weights from Drive on restart, skips retraining
- ✅ **Hybrid RAG Search** — 60% semantic (ChromaDB) + 40% keyword (BM25) for precise medical retrieval
- ✅ **Multi-book Indexing** — Felson\'s Chest Roentgenology + ESR Chest Imaging Guide
- ✅ **Grad-CAM Visualization** — heatmaps showing where the model looks
- ✅ **Structured Clinical Report** — 5-section report: Impression → Findings → Differential → Next Steps → AI Caveat
- ✅ **LLM Agnostic** — supports Gemini 1.5 Pro, GPT-4o, or local Ollama Llama3
- ✅ **Custom Image Input** — upload any X-ray via Colab file picker
- ✅ **Full Drive Persistence** — weights, ChromaDB, and reports all saved to Google Drive

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PHASE 1 & 2 — Training                │
│                                                         │
│  Dataset (Kaggle)                                       │
│      ↓  Train/Val/Test Split                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐          │
│  │ ResNet50 │  │ResNet101 │  │ DenseNet121  │          │
│  │  (30%)   │  │  (30%)   │  │    (40%)     │          │
│  └──────────┘  └──────────┘  └──────────────┘          │
│         ↓ Weighted Softmax Ensemble                     │
│     Ensemble Acc: ~95.7%  |  Macro F1: ~95.5%          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  PHASE 4 — Book Indexing                │
│                                                         │
│  Felson\'s Chest Roentgenology (PDF)                    │
│  ESR Chest Imaging Guide (PDF)                          │
│      ↓  PyMuPDF text extraction                         │
│      ↓  RecursiveCharacterTextSplitter (800 chars)      │
│      ↓  all-MiniLM-L6-v2 Embeddings                    │
│      ↓  ChromaDB Vector Store (persisted to Drive)      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                PHASE 5 — RAG + LLM Inference            │
│                                                         │
│  Input X-Ray                                            │
│      ↓  Ensemble → diagnosis + confidence               │
│      ↓  Hybrid Search (ChromaDB 60% + BM25 40%)         │
│      ↓  Top 5 book chunks retrieved                     │
│      ↓  LLM Prompt (diagnosis + context)                │
│      ↓  5-Section Clinical Report                       │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Results

| Model | Val Acc | Test Acc | Macro F1 |
|---|---|---|---|
| ResNet50 | 91.2% | 90.8% | 90.5% |
| ResNet101 | 92.8% | 92.1% | 91.9% |
| DenseNet121 | 94.1% | 93.7% | 93.4% |
| 🤝 **Ensemble** | **95.8%** | **95.7%** | **95.5%** |

---

## 🗂️ Project Structure

```
chest_xray_project/                  ← Google Drive root
│
├── dataset/
│   ├── train/
│   │   ├── COVID/
│   │   ├── Lung_Opacity/
│   │   ├── Normal/
│   │   └── Viral Pneumonia/
│   ├── val/
│   └── test/
│
├── weights/
│   ├── ResNet50_checkpoint.pth
│   ├── ResNet101_checkpoint.pth
│   ├── DenseNet121_checkpoint.pth
│   ├── model_meta.json
│   └── ensemble_config.json
│
├── reports/
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── ensemble_confusion_report.png
│   ├── gradcam_visualizations.png
│   ├── pipeline_demo.png
│   └── custom_image_report.png
│
├── books/
│   ├── felsons_chest_xray.pdf       ← add your PDF here
│   └── esr_chest_imaging.pdf        ← add your PDF here
│
└── chroma_db/                       ← auto-created by Cell 22
    └── (ChromaDB vector files)
```

---

## ⚙️ Setup

### 1. Clone & Open in Colab

```bash
git clone https://github.com/yourusername/chest_xray_project.git
```

Open `Chest_XRay_AI.ipynb` in [Google Colab](https://colab.research.google.com/) with a **GPU runtime** (T4 recommended).

### 2. Add Medical PDFs to Drive

Place these in `MyDrive/chest_xray_project/books/`:

| PDF | Where to get |
|---|---|
| `felsons_chest_xray.pdf` | Elsevier / your library |
| `esr_chest_imaging.pdf` | Free at [myesr.org](https://www.myesr.org) |

### 3. Get a Gemini API Key (Free)

```
1. Go to → aistudio.google.com
2. Sign in → Get API Key → Create API key
3. Paste into Cell 25:
   os.environ["GOOGLE_API_KEY"] = "AIza..."
```

### 4. Run All Cells

```
Cell 1  → 27  (first time: ~2-3 hours with training)
Cell 1  → 27  (subsequent sessions: ~5 minutes, loads from Drive)
```

---

## 🚀 Usage

### Classify Your Own X-Ray

Run **Cell 28** in the notebook:

```python
# Opens file picker → displays image → generates full report
from google.colab import files
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
result   = generate_clinical_report(img_path)
```

### Sample Clinical Report Output

```
══════════════════════════════════════════════════════════════════════
           🏥  AI CLINICAL RADIOLOGY REPORT
══════════════════════════════════════════════════════════════════════
  🩺 Diagnosis  : COVID-19  (94.2% confidence)
  📚 Sources    : felsons_chest_xray.pdf, esr_chest_imaging.pdf
──────────────────────────────────────────────────────────────────────

## 1. CLINICAL IMPRESSION
Chest radiograph demonstrates bilateral peripheral ground-glass
opacification consistent with COVID-19 pneumonia.

## 2. RADIOLOGICAL FINDINGS
• Bilateral peripheral ground-glass opacity (GGO)
• Lower lobe predominant distribution
• Absence of pleural effusion
• No significant lymphadenopathy

## 3. DIFFERENTIAL DIAGNOSIS
• Viral Pneumonia — similar interstitial pattern but typically
  unilateral; lacks peripheral predominance
• Lung Opacity — consolidation rather than GGO pattern

## 4. CLINICAL CORRELATION & NEXT STEPS
RT-PCR testing recommended to confirm COVID-19 diagnosis.
CT chest for detailed characterisation if clinically indicated.

## 5. AI CONFIDENCE ASSESSMENT
Model confidence: 94.2%. This is an AI-assisted report and must
be interpreted by a qualified radiologist before clinical use.
══════════════════════════════════════════════════════════════════════
```

---

## 📓 Notebook Cell Map

| Phase | Cell | Description |
|---|---|---|
| **0 — Config** | 1 | Global paths, constants, folder creation |
| **1 — Data** | 2 | Download dataset from Kaggle |
| | 3 | DataLoaders + augmentation |
| | 4 | Visualise sample batch |
| **2 — Training** | 5 | Model definitions (ResNet50/101, DenseNet121) |
| | 6 | Training & evaluation functions |
| | 7 | Train all 3 models (smart load-or-train) |
| | 8 | Final accuracy summary |
| | 9 | Training curves |
| | 10 | Confusion matrices |
| | 11 | Per-class classification reports |
| | 12 | Model comparison bar chart |
| | 13 | Save best model |
| | 14 | Inference demo |
| **3 — Ensemble** | 15 | Ensemble setup + weights |
| | 16 | Ensemble evaluation vs individual models |
| | 17 | Ensemble confusion matrix + report |
| | 18 | Save ensemble config |
| **4 — Book Indexing** | 19 | Install RAG dependencies |
| | 20 | Load medical PDFs from Drive |
| | 21 | Semantic chunker (800 char chunks) |
| | 22 | Embed → store in ChromaDB |
| | 23 | Test retrieval — all 4 classes |
| **5 — RAG + LLM** | 24 | Hybrid search (BM25 + semantic) |
| | 25 | LLM setup (Gemini / GPT-4o / Ollama) |
| | 26 | Clinical report generator function |
| | 27 | Full end-to-end pipeline demo |
| **Custom Input** | 28 | Upload your own X-ray → full report |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Deep Learning** | PyTorch, torchvision |
| **Models** | ResNet50, ResNet101, DenseNet121 (ImageNet pretrained) |
| **Explainability** | Grad-CAM |
| **RAG Framework** | LangChain |
| **Vector Database** | ChromaDB |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Keyword Search** | BM25 (rank-bm25) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **LLM** | Gemini 1.5 Pro / GPT-4o / Ollama Llama3 |
| **Environment** | Google Colab (T4 GPU) |
| **Storage** | Google Drive |

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**.
> The AI-generated clinical reports must **not** be used as a substitute
> for diagnosis by a qualified medical professional.
> Always consult a licensed radiologist for clinical decisions.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ using PyTorch + LangChain + Gemini

⭐ Star this repo if you found it useful!

</div>
