# Kohonen SOM challenge

Written answers are in the **Challenge** section of `kohonen.ipynb`, including a vectorised `train` that is much faster than the nested-loop version.

The last question is about productionising this. I built a small serverless demo rather than only describing it:

- Lambda runs training
- Cognito for login
- CloudFront + S3 for the page and result images

Live site: https://d2go0fq5y1eouk.cloudfront.net/  
Login details sent separately.

The app has two modes. One is the original RGB example (random 3-d points, map shown as colour). The other is a 128-d CIFAR-10 photo atlas so the same trainer can be seen on image embeddings. Each run shows the untrained map next to the trained one.

Code for the demo is in `production_app/`. The SOM itself is `production_app/kohonen/som.py`.
