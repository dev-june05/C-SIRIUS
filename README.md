# Cross-Modal Satellite Image Retrieval (PS11)

This repository contains the setup for the ISRO Bharatiya Antariksh Hackathon Problem Statement 11. It implements a dual-encoder architecture for cross-modal (SAR to Optical) satellite image retrieval.

## Project Structure
- `src/data/dataset.py`: PyTorch Dataset for loading and preprocessing SAR (dB conversion, clip, norm) and Optical (min-max norm) images.
- `src/models/dual_encoder.py`: The ResNet-50 (SAR) and ViT-B (Optical) encoders with 512-dimensional projection heads.
- `requirements.txt`: Python dependencies.

## Teammate Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dev-june05/C-SIRIUS/
   cd C-SIRIUS
