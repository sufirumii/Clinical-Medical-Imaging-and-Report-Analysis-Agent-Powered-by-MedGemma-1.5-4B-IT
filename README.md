# Clinical Medical Imaging and Report Analysis Agent Powered by MedGemma 1.5 4B IT

A production-grade multimodal medical AI agent for clinical image interpretation, report generation, and diagnostic reasoning, powered by Google MedGemma 1.5 4B.

---

## Overview

This is a multi-file, modular medical AI system built on top of Google's MedGemma 1.5 4B vision-language model. It is designed to assist clinicians and researchers with the interpretation of medical images and clinical documents through natural language interaction. The system is not a single-script demo — it separates model inference, API serving, and the frontend interface into distinct, independently maintainable layers.

MedGemma 1.5 is trained on a diverse corpus of de-identified medical data including radiology images, histopathology slides, dermatology images, ophthalmology fundus images, and electronic health records. It supports a 128K token context window and significantly outperforms general-purpose vision-language models on clinical benchmarks.

---

## Capabilities

- Chest X-ray interpretation and structured radiology report generation
- CT and MRI image analysis including 3D volumetric reasoning
- Dermatology image classification and lesion assessment
- Histopathology slide interpretation
- Fundus image analysis for diabetic retinopathy staging
- Differential diagnosis generation with clinical reasoning
- Severity and urgency assessment from imaging findings
- Anatomical structure localization and bounding box detection
- Lab report parsing and structured data extraction from unstructured documents
- Electronic health record comprehension and question answering
- Longitudinal imaging comparison between current and prior scans

---

## Architecture

```
MedSentinel/
├── model.py          # Model loading and inference logic (MedGemma 1.5 4B)
├── api.py            # Gradio frontend and session management
├── uploads/          # Temporary storage for uploaded medical images
├── requirements.txt  # Python dependencies
└── README.md
```

The system is intentionally decoupled:

- **model.py** handles all interaction with the MedGemma model, including processor initialization, chat template formatting, and inference. This layer can be extended to support batching, quantization, or alternative model backends without touching the API or frontend.
- **api.py** handles the user interface and session state. It manages multi-turn conversation history and passes image and text inputs to the model layer.
- The architecture is designed to be extended with a FastAPI REST backend, FHIR integration, retrieval-augmented generation over clinical knowledge bases, or multi-agent orchestration.

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU with at least 16 GB VRAM (tested on RTX 5000 QuadroIN1)
- Hugging Face account with access granted to google/medgemma-1.5-4b-it
- Acceptance of the Health AI Developer Foundations terms of use

---

## Installation

```bash
git clone https://github.com/sufirumii/Clinical-Medical-Imaging-and-Report-Analysis-Agent-Powered-by-MedGemma-1.5-4B-IT.git
cd clinical-medgemma-agent
pip install transformers>=4.50.0 accelerate torch Pillow requests huggingface_hub fastapi uvicorn pydantic python-multipart gradio
```

Authenticate with Hugging Face:

```bash
huggingface-cli login
```

---

## Usage

```bash
python api.py
```

This will load the MedGemma 1.5 4B model and launch the Gradio interface with a public share link. On first run, model weights (approximately 9 GB) will be downloaded from the Hugging Face Hub.

Once running, you can:

- Upload a medical image using the image panel
- Type a clinical question or select a quick prompt
- Receive a structured natural language analysis from MedGemma

---

## Example Prompts

- "Describe this chest X-ray in detail including all visible findings"
- "Generate a complete structured radiology report for this CT scan"
- "Suggest the top differential diagnoses with clinical reasoning"
- "Summarize this lab report and flag all abnormal values"
- "Assess the severity and urgency of the findings in this image"
- "Compare this scan with the prior imaging and describe any interval changes"

---

## Model Information

| Property | Value |
|---|---|
| Model | google/medgemma-1.5-4b-it |
| Parameters | 4 billion |
| Modalities | Text, medical images |
| Context window | 128K tokens |
| Precision | bfloat16 |
| License | Health AI Developer Foundations Terms |

MedGemma 1.5 4B outperforms MedGemma 1 4B on 3D radiology classification, whole-slide histopathology, fundus imaging, EHR question answering, and lab document extraction. Full benchmark results are available in the MedGemma Technical Report (arXiv:2507.05201).

---

## Limitations

- MedGemma 1.5 is intended as a research and development starting point. It is not validated for direct clinical use and should not be used to inform patient care decisions without independent clinical review.
- Inference on a single GPU may take 60 to 120 seconds per query depending on image complexity and available VRAM.
- The model has been primarily evaluated on English-language prompts.
- Performance varies across imaging modalities. Fine-tuning on domain-specific data is recommended for production deployment.

---

## Roadmap

- FastAPI REST backend for programmatic access
- FHIR R4 integration for EHR connectivity
- Retrieval-augmented generation over clinical guidelines and literature
- Quantized model support for lower VRAM environments
- Multi-turn clinical reasoning with persistent patient context
- DICOM file support for direct radiology workflow integration

---

## Citation

If you use MedGemma in your work, please cite:

```
@article{sellergren2025medgemma,
  title={MedGemma Technical Report},
  author={Sellergren, Andrew and Kazemzadeh, Sahar and others},
  journal={arXiv preprint arXiv:2507.05201},
  year={2025}
}
```

---

## License

This project is released under the MIT License. Use of the underlying MedGemma model is governed by the Health AI Developer Foundations terms of use. This system is intended for research purposes only and is not approved for clinical diagnostic use.
