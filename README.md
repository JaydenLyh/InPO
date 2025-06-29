# InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment (CVPR 2025 Highlight) 
[arXiv Paper](http://arxiv.org/abs/2503.18454) | [Project Page](https://jaydenlyh.github.io/InPO-project-page/) | [Poster](https://cvpr.thecvf.com/virtual/2025/poster/33603) | [SmPO-Diffusion Project Page](https://jaydenlyh.github.io/SmPO-project-page/)

![photo](./assets/inpo.png "InPO-SDXL")

The repository provides the official implementation, experiment code, and model checkpoints used in our research paper.

---

## 📖 News & Updates
- **[2025-03-24]** 🎉 Preprint paper released on arXiv!
- **[2025-03-24]** ✅ Initial model checkpoints published
- **[2025-06-04]** 📊 Poster and project page
- **[2025-06-29]** 🚀 Training code release

---

## 🌟 Key Features
- Preservation of Pre-Trained Knowledge
- Targeted Performance Improvement
- Optimized Computational Efficiency
- Interpretability and Parametric Control
- Cross-Domain Broader Applicability

---

## 🔧 Quick Start

### Installation 
```bash
conda create -n ddiminpo python=3.10
conda activate ddiminpo
git clone https://github.com/JaydenLyh/InPO.git
cd InPO
pip install -r requirements.txt
```
---
### Preparation of dataset and base models
```bash
InPO/
├── assets/                   
│   └── inpo.png            
├── checkpoints/            
│   ├── stable-diffusion-v1-5/          
│   ├── sdxl-vae-fp16-fix/            
│   └── stable-diffusion-xl-base-1.0/         
├── datasets/                 
│   └── pickapic_v2/   
├── train.py            
├── README.md              
├── LICENSE.txt            
└── requirements.txt       
```
---
### Training for SDXL
```bash
export MODEL_NAME="checkpoints/stable-diffusion-xl-base-1.0"
export VAE="checkpoints/sdxl-vae-fp16-fix"
export DATASET_NAME="pickapic_v2"
PORT=$((20000 + RANDOM % 10000))

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port $PORT --mixed_precision="fp16" --num_processes=8 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=400 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 50 \
  --beta_dpo 5000 \
  --sdxl  \
  --output_dir="ddiminpo-sdxl" 
```
---


## Our Models
| Model          | Download Links                          
|----------------|-----------------------------------------|
| InPO-SD1.5     | [Hugging Face](https://huggingface.co/JaydenLu666/InPO-SD1.5)  |
| InPO-SDXL    |  [Hugging Face](https://huggingface.co/JaydenLu666/InPO-SDXL)       |


## Citation
```bash
@inproceedings{lu2025inpo,
  title={InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment},
  author={Lu, Yunhong and Wang, Qichao and Cao, Hengyuan and Wang, Xierui and Xu, Xiaoyin and Zhang, Min},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={28629--28639},
  year={2025}
}
```

## Acknowledgments
The implementation of this project references the [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) repository by Salesforce AI Research. We acknowledge and appreciate their open-source contribution.