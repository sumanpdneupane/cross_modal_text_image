# Cross-Modal Recipe Retrieval Using Images and Text Descriptions with Transfer Learning

# Research papers
```
Direct Links (Useful)
Recipe1M+ project page: 
    https://im2recipe.csail.mit.edu/
Recipe1M+ paper: 
    https://arxiv.org/abs/1810.06553
```

# Data sets
```
https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images
```

# Slides
```
```

# Installation
```
pip install -r requirements.txt

or

pip3 install -r requirements.txt
```

# Run Project
```
1. Install requirements.txt
2. Go To main.ipynb to run the project
    a. This function will automatically detect the machine type 
       and run either in GPU i.e. CUPD, Mac i.e. MPS
       or CPU
       DEVICE = get_device()
    b. BATCH_SIZE = 16, Change it according to your device performance.
```

# Project Overview
```
This research presents the design, implementation, and evaluation of a cross-modal recipe 
retrieval system that aligns food images and recipe text within a shared semantic embedding 
space. The proposed system supports bidirectional retrieval, enabling both image-to-recipe 
and recipe-to-image queries using transfer learning and contrastive learning. The core 
contribution is a multi-encoder architecture that explicitly models semantic differences 
between visual data, short textual descriptions, and long-form procedural text.

A pretrained ResNet50 model is used as the visual encoder to extract high-level semantic 
features from food images. For textual representation, two separate DistilBERT-based encoders 
are employed: one for short recipe titles and another for long-form ingredient lists and cooking 
instructions. This separation preserves fine-grained procedural information without overwhelming 
the shared embedding space. Outputs from each encoder are projected into a unified embedding 
space using modality-specific alignment modules composed of linear layers, GELU activation, 
layer normalization, and dropout.

Training is performed using the InfoNCE contrastive loss, optimizing relative similarity between 
matched image-text pairs while pushing apart mismatched pairs within each batch. A multi-pair 
contrastive strategy computes six contrastive losses across all modality combinations and averages 
them to ensure balanced alignment. Transfer learning reduces computational cost and enables efficient 
training under limited hardware constraints.

Experiments are conducted on a publicly available Kaggle food image-recipe dataset containing 
approximately 13,500 paired samples. Performance is evaluated using Recall@K metrics 
(R@1, R@5, and R@10) for all retrieval directions. Results show strong retrieval performance, 
with Recall@10 exceeding 0.99 for both image-to-text and text-to-image tasks. Compared to the 
Im2Recipe baseline (R@10 ≈ 0.65), the proposed approach demonstrates substantial improvement. 
Overall, the research confirms that contrastive learning and specialized multimodal encoders 
can produce effective cross-modal retrieval systems even with limited resources.
```
# Architecture
![Architecture](assets/architecture.png)

# Major Outcomes
### Training and validation Summary
![Training Loss Summary](assets/training_and_validation_summary.png)

### Training and validation Curve
![Training Loss Graph](assets/training_and_validation_graph.png)

### Rcall@k Sumarry
![Comparision](assets/r@call_summary.png)

### Comparision with research paper
![Comparision](assets/comparision_with_research_paper.png)


# Tested Model Results
### Outcome title -> ingredients_instructions
![Outcome title -> ingredients_instructions](assets/1.png)

### Outcome image -> image
![Outcome imageimage](assets/2.png)

### Outcome image -> title
![Outcome image -> title](assets/3.png)

### Outcome image -> ingredients_instructions
![Outcome image -> ingredients_instructions](assets/4.png)


# Instructions to run inference using the provided .pt / .pth model file
```
1. You can find fusion_model.pth file above main.ipynb
2. Open main.ipynb file
    a. Run Hyperparameters, Dataset and DataLoader and Initializations codes
    b. Then goto  Test Model section
       - Run Dataset Embeddings
       - Then you can run and inference 'Retrieve from your query' section 
         to visualize data.
```