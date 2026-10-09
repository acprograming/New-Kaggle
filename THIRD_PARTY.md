# Source acknowledgements

`notebooks/RSNA_Knee_Public950_Improved.ipynb` and its qualification derivative adapt the public DINOsaur notebook by romantamrazov, immutable Kaggle version 356806064, observed as Apache 2.0 on Kaggle:
https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5

The public ConvNeXt-MIL reader is by goodpjw2008. The anatomical mirror/SWA reader is by nartaa, with Raptor and supporting inference contributions from dreaddevelopment, Mattia Angeli, Jiwei Liu, sujanmajhisuzan, and evgendvorkin. Their model/data acknowledgements remain in the notebooks, including Osteoarthritis Initiative controlled-access data, NDA Study 3407, DOI 10.15154/0hcg-f676. Attached datasets and pretrained models retain their original terms.

The original spatial experiment extends our earlier TargetMIL training code. Its zero-initialized per-finding spatial query experiment is informed by the public Helios training source, immutable Kaggle version 356507041. It does not reproduce or include the private Helios model bundle:
https://www.kaggle.com/code/heliosli/rsna-knee-abnormality-detection

The ConvNeXt implementation comes from timm / PyTorch Image Models (Apache 2.0):
https://github.com/huggingface/pytorch-image-models

Further methodological reading:
https://github.com/mathischmp/challenge_rsna

No competition MRI images, radiology reports, subject identifiers, or label tables are included in this repository.
