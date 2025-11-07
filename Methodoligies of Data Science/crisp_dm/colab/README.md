# CRISP-DM Colab Version

## Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/data-mining-methodologies-portfolio/blob/main/crisp_dm/colab/CRISP_DM_colab.ipynb)

### Setup Steps

1. **Upload Kaggle API Key** (if not already done):
   ```python
   from google.colab import files
   files.upload()  # Upload kaggle.json
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Run All Cells** - The notebook will:
   - Install dependencies
   - Download Rossmann dataset (2GB)
   - Run all 6 CRISP-DM phases (~20 min)

3. **Save Results** - Mount Google Drive to persist models:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### Differences from Local Version

- **No `src/` folder**: All functions inlined in notebook
- **Data Path**: Uses `/content/rossmann/` instead of `../data/`
- **Model Save**: Defaults to `/content/` (ephemeral unless Drive mounted)

### Resources

- [CRISP-DM Master Prompt](../prompts/00_master_prompt.md)
- [Dr. Provost Critic Persona](../prompts/critic_persona.md)
- [Business Report](../reports/business_understanding.md)
