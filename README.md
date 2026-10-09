# RSNA Knee experiments

This repository holds the reproducible public inference baseline and original experiments for RSNA Knee Abnormality Detection.

The upstream public recipe scored 0.950. Our best measured submission is 0.943; our derivative and new experiments have not earned a new competition score yet.

Work in progress: controlled spatial-attention training, an exploratory combiner trained on earlier out-of-fold predictions, and checks that reject incomplete or invalid submissions. Code and aggregate results will be recorded here. Competition MRI images, reports, and labels are supplied separately at execution time.

Current compute: this workspace has no GPU. Kaggle browser access is timing out. Standard public GitHub Actions runners can test code on CPU; a GPU and mounted MRI data are required for full image-model training.
