# DRAIN
GitHub Repo for ICLR 2023 (Oral) Paper *"Temporal Domain Generalization with Drift-Aware Dynamic Neural Networks"*

<img src="./model_architecture.PNG" width="790" height="290">

Our experiments include both classificaiton and regression datasets. For example, to run our experiments on 2-Moons dataset, go to the "classification" folder and do

1. name model_moons.py as model.py

2. python train.py --dataset Moons

Similar process for other datasets.

The code has been tested with PyTorch and Anaconda.



If you find this code useful in your research, please consider citing:

@article{bai2022temporal,
  title={Temporal Domain Generalization with Drift-Aware Dynamic Neural Networks},
  author={Bai, Guangji and Ling, Chen and Zhao, Liang},
  journal={arXiv preprint arXiv:2205.10664},
  year={2022}
}
