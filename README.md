# Black-Hole Driven Identity Absorbing  (CVPR 2025)

<p align="left">
   <a href="https://your-website-link.com">Muhammad Shaheryar*</a>,
  <a href="https://co-author1.com">Jong Taek Lee</a>,
  <a href="https://co-author2.com">Soon Ki Jung</a>,
  School of Computer Science & Engineering, Kyungpook National University  
  <br><br>
  <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Shaheryar_Black_Hole-Driven_Identity_Absorbing_in_Diffusion_Models_CVPR_2025_paper.html">
    <img src="https://img.shields.io/badge/Paper-4CAF50?style=for-the-badge" />
  </a>

  <!-- Supplementary -->
  <a href="https://openaccess.thecvf.com/content/CVPR2025/supplemental/Shaheryar_Black_Hole-Driven_Identity_CVPR_2025_supplemental.pdf">
    <img src="https://img.shields.io/badge/Supplementary-66CCFF?style=for-the-badge" />
  </a>

</p>


## Overview  

Black Hole-Driven Identity Absorption (BIA), a novel approach for identity erasure within the latent space of diffusion models. BIA uses a “black hole” metaphor, where the latent region representing a specified identity acts as an attractor, drawing in nearby latent points of surrounding identities to “wrap” the black hole. Instead of relying on randomtraversals for optimization, BIA employs an identity absorption mechanism by attracting and wrapping nearby validated latent points associated with other identities to achieve a vanishing effect for specified identity. Our method effectively prevents the generation of a specified identity while preserving other attributes. 
<p align="center">
<img width="500" height="250" alt="supp_2" src="https://github.com/user-attachments/assets/4c744ff8-b648-4264-9431-e4bf92f1833a" />
</p>

## Latest Updates  
- 2025-10-10: We released the code, model, dataset, and evaluation scripts.  
- 2025-02-27: Paper accepted.  

---




## Environment setup

```bash
# Create and activate a conda environment
conda create --name bia python=3.8
source activate bia

# Install PyTorch + CUDA (adjust based on your GPU / CUDA version)
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Clone the repo and install requirements
git clone https://github.com/mshaheryar91/BlA.git
cd BIA
```

---
## Black Hole Region Formation
We adopt the linear SVM to search for the black hole on different latent space levels. To form the black hole, we need images with identity annotations. We use the CelebA-HQ-256 dataset as an example, in which the dataset contains the annotations for different identities. The following command form the identity attribute, which we want to unlearn and use the real images from CelabA-HQ to search for the blackhole, and automatically stores two hyperplanes in the folder ./boundary/.

```bash
python main.py --bh_boundary --exp ./runs/test \
    --t_0 500 \
    --n_inv_step 40 \
    --n_test_step 40 \
    --n_iter 2 \
    --img_path imgs/celeb2.png \
    --boundary_h 'path/to/target_boundary_h' \
    --boundary_z 'path/to/target_boundary_z' \
    --start_distance -100 \
    --end_distance 100 \
    --edit_img_number 20
```
Note: To from a good and well-define black hole, we may need to adjust the number of image samples used according to different identites. An indicator for the black hole is the classification accuracy for the test split.
Pre-localized boundaries: We provide some pre-localized boundaries for easy use.

<img width="3746" height="1106" alt="black_hole_finalday" src="https://github.com/user-attachments/assets/13533344-1803-4515-a95f-9c992e950a20" />

---
## 📊 Results on the CelebA-HQ dataset showcasing different identities.
<img width="2807" height="1253" alt="Idsall" src="https://github.com/user-attachments/assets/86f52836-12d2-400f-9e0e-e0ef5811103e" />

-----
## 📊 Quantitative Results compared with SOTA.

<img width="1209" height="263" alt="image" src="https://github.com/user-attachments/assets/c48da3d2-05fb-4a23-ad54-4eea81371230" />

---
## 📜 Citation
If you find our work interesting and useful, please consider citing it.
```bash
@InProceedings{Shaheryar_2025_CVPR,
    author    = {Shaheryar, Muhammad and Lee, Jong Taek and Jung, Soon Ki},
    title     = {Black Hole-Driven Identity Absorbing in Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {28544-28554}
}
```
