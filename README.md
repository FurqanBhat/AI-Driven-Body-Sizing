# AI-Driven-Body-Sizing

Predict 14 human body measurements from front and side silhouette images, combined with basic inputs like height and weight, using a deep learning pipeline based on EfficientNetV2.

---

##  Overview

This project presents a deep learning model for estimating 14 key human body measurements using:
- Front and side silhouette images
- Height and weight maps

The model is trained on the BodyM dataset and achieves an **MAE of ~1.40 cm** on unseen, real-world images. Ideal for virtual try-on, fitness tracking, and ergonomic applications.

---

##  Model Architecture

- Backbone: `EfficientNetV2-S` (from `timm`, `tf_efficientnetv2_s.in21k`)
- Input: 4 channels → `[front_img, side_img, height_map, weight_map]`
- Regression head: Two Linear+ReLU+Dropout blocks with residual skip connection
- Output: 14 continuous body measurements
- Trainable Parameters: ~5.3M

---



