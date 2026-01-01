# Cross-Modal Recipe Retrieval Using Images and Text Descriptions with Transfer Learning

# Turorials
```
How do Multimodal AI models work? Simple explanation
https://www.youtube.com/watch?v=WkoytlA3MoQ

Coding Stable Diffusion from scratch in PyTorch
https://www.youtube.com/watch?v=ZBKpAp_6TGI
https://github.com/hkproj/pytorch-stable-diffusion

Convolution Visualizer
https://ezyang.github.io/convolution-visualizer/


New
https://www.youtube.com/watch?v=6a-qMRguYE8

OpenAI CLIP: ConnectingText and Images (Paper Explained)
https://www.youtube.com/watch?v=T9XSU0pKX2E&t=219s


Text-Image Retrieval and Matching in pytorch
https://youtu.be/eiDBpmAcpik

```

# Research papers
```
Direct Links (Useful)
Recipe1M+ project page: 
    https://im2recipe.csail.mit.edu/
Recipe1M+ paper: 
    https://arxiv.org/abs/1810.06553
GitHub code: 
    https://github.com/torralba-lab/im2recipe-Pytorch
    http://github.com/torralba-lab/im2recipe-Pytorch?utm_source=chatgpt.com

https://chatgpt.com/share/69443660-5548-800d-99cc-977551d9de73
https://chatgpt.com/c/694e6b89-23d0-8322-bb7d-6e5611ab1fe9
```

# Data sets
```
https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images
```

```angular2html
What you SHOULD say academically

If this is for a project report / thesis / viva, say this:
“For cross-modal image–text retrieval, fusion is not required. 
We use a dual-encoder architecture where image and text are encoded independently 
and aligned into a shared embedding space using contrastive learning. This design 
enables efficient and symmetric image-to-text and text-to-image retrieval.”

This is 100% correct and aligns with literature.
```


```angular2html
nomic: https://ollama.com/library/nomic-embed-text
all mini lm: https://ollama.com/library/all-minilm:l6-v2
```

```angular2html

# Contrastive Learning
https://www.youtube.com/watch?v=u-X_nZRsn5M&list=PLV8yxwGOxvvreie4ioSYlfg1md4_OJz-q

# Embeddings
https://www.youtube.com/watch?v=mCvW_qNm7rY&list=PLhmu2E7RVPVDgdoKYAxmy44IGwqKmgTNY

# Modifying BERT
https://www.youtube.com/watch?v=wVrHPZz5jic

```

https://github.com/gorjanradevski/cross_modal_full_transfer/blob/master/src/models.py
https://www.youtube.com/watch?v=etjz4Hy0fzA
https://www.youtube.com/watch?v=nZ5j289WN8g


Learn SBERT Sentence Transformers: TSDAE, SimCSE and CT
https://www.youtube.com/watch?v=6yPWtdgs5Sg


Build Semantic Search Engine with S-BERT
https://www.youtube.com/watch?v=Yo4NqGPISXQ&t=225s
https://www.sbert.net/docs/sentence_transformer/pretrained_models.html


https://www.google.com/search?q=cross+model+bert+and+resnet50+retrieval+system+pytorch+code&sca_esv=6c5b9535e4a97562&rlz=1C5CHFA_enNP1094NP1094&sxsrf=AE3TifOgxVVmKCn-raOICFQ1VB1GvI5TNQ%3A1767027891271&ei=s7RSaaiiEJCq0-kPjePc8AQ&oq=cross+model+bert+and+resnet50+retrieval+system+pytorch+&gs_lp=Egxnd3Mtd2l6LXNlcnAiN2Nyb3NzIG1vZGVsIGJlcnQgYW5kIHJlc25ldDUwIHJldHJpZXZhbCBzeXN0ZW0gcHl0b3JjaCAqAggCMgUQIRigATIFECEYoAEyBRAhGKABMgUQIRigATIFECEYoAEyBRAhGJ8FMgUQIRifBTIFECEYnwUyBRAhGJ8FSPouUNAEWIYfcAF4AJABAJgB3QKgAcIGqgEHMC4zLjAuMbgBA8gBAPgBAZgCBaAC4QbCAg4QABiABBiwAxiGAxiKBZgDAIgGAZAGAZIHBzEuMy4wLjGgB-4bsgcHMC4zLjAuMbgHzgbCBwMxLjTIBweACAA&sclient=gws-wiz-serp


```angular2html
Late Fusion / Similarity-Based
Compute embeddings separately (which you already have).
Then calculate a similarity score (e.g., cosine similarity) between image and text embeddings
This is simpler and often used in retrieval systems.

Early Fusion / Joint Embedding
Combine embeddings before further processing, e.g., concatenate [image_embed || text_embed] and feed into another network for classification or scoring.
Can also use attention-based fusion (cross-attention between modalities).

Hybrid / Multi-Modal Fusion
Weighted combination, gated fusion, or transformer-based fusion layers that consider interactions between the modalities.
```

miso-butter-roast-chicken-acorn-squash-panzanella

"miso butter roast chicken with acorn squash"