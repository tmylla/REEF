
<!-- <p align="center">
  <img src="assets/logo.png"  height=120>
</p> -->


## <div align="center">REEF: Representation Encoding Fingerprints for Large Language Models<div> 

<div align="center">
<a href="https://github.com/tmylla/Persafety"><img src="https://img.shields.io/static/v1?label=Github&message=Arxiv:REEF&color=red&logo=arxiv"></a> &ensp;
</div>



## 🌈 Introduction
We are excited to present 'REEF: Representation Encoding Fingerprints for Large Language Models,' an efficient and robust approach designed to protect the intellectual property of open-source LLMs. 

> In this paper, we propose a training-free REEF to identify the relationship between the suspect and victim models from the perspective of LLMs' feature representations. Specifically, REEF computes and compares the centered kernel alignment similarity between the representations of a suspect model and a victim model on the same samples. This training-free REEF does not impair the model's general capabilities and is robust to sequential fine-tuning, pruning, model merging, and permutations.

In summary, REEF provides a simple and effective way for third parties and model owners to protect LLMs' intellectual property together.


![Overview Diagram](assets/overview.png)




<!-- ## 🚀Getting Started

### 💻Prerequisites

### 🔧Installation

### 🌟Usage -->


## 💪To-Do List
We are currently organizing the code for Persafety. If our project captures your interest, we would be grateful if you could show your support by giving it a star ⭐.

## 📝License
Distributed under the Apache-2.0 License. See LICENSE for more information.

<!-- ## 📖BibTeX
```
todo
``` -->


<!-- 0. Obtain Activations

```Bash
generate_activation.py

# call using save_activation.sh
```

1. DNN Classifier (Section 3)

```Bash
train_cls.py  # train linear/MLP/CNN classifier

train_cls_gcn.py  # train GCN classifier

transfer_cls.py  # apply a classifier to suspect models
```

2. REEF (Section 5)

```Bash
compute_cka.py  # CKA similarity within REEF

# Reproduction Code of "Human-Readable Fingerprint for Large Language Models"
pcs.py
ics.py

# Reproduction Code of "A Fingerprint for Large Language Models"
generate_head_activations.py
logit.py

# Evade REEF with Fine-tuning
evade-reef.py

``` -->



