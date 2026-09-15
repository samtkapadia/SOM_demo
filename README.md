# Kohonen SOM challenge

Written answers are in the "Challenge" section of `kohonen.ipynb`, including a vectorised `train` that is much faster than the nested-loop version.

The last question is about productionising the SOM. Here I've deployed in a small serverless demo to accompany my described approach:

- Lambda runs training
- Cognito for login
- CloudFront + S3 for the page and result images

Live site: https://d2go0fq5y1eouk.cloudfront.net/ (login details sent separately).

The app has two modes. One is the original RGB example (random 3-d points, map shown as colour). The other is a 128-d CIFAR-10 photo atlas to demonstrate my implementation's capacity to generalise to embeddings with more than 3 features. Each run shows the untrained map next to the trained one.

The atlas embeddings are built offline, not in Lambda. `scripts/embed_cifar.py` takes 25 test images per CIFAR-10 class, runs a frozen ImageNet ResNet-18 (512-d avgpool), L2-normalises, then PCA down to 128-d. The embeddings are stored in an `.npz` on S3 (`atlas/embeddings.npz`).

I chose to keep the feature extractor off Lambda because it only needs to run once, and PyTorch plus scikit-learn would make the function much heavier than the SOM itself. For the purposes of this demo, my intention was that clicking Train should just organise a fixed set of 128-d vectors and not worry about re-embedding CIFAR.

Code for the demo is in `production_app/`. The SOM itself is `production_app/kohonen/som.py`.
