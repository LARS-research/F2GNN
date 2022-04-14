# F2GNN -- Designing the Topology of Graph Neural Networks: A Novel Feature Fusion Perspective
#### This repository is the code for our WWW 2022 paper: [Designing the Topology of Graph Neural Networks: A Novel Feature Fusion Perspective](https://arxiv.org/pdf/2112.14531.pdf)

#### Overview
In this paper, we provide a novel feature fusion perspective in designing the GNN topology.
Firstly, we propose a novel framework to unify the existing topology designs with feature selection and fusion strategies. 
It transforms the GNN topology design into the design of this two strategies.
Then we develop a NAS method on top of the unified framework containing a novel search space and an improved differentiable search algorithm.
Extensive experiments on eight real-world datasets demonstrate that the proposed **F2GNN** can improve performance while alleviating the deficiencies, especially alleviating the over-smoothing problem.

#### Requirements

     torch-cluster==1.5.7
     torch-geometric==1.7.2
     torch-scatter==2.0.6
     torch==1.6.0
     numpy==1.17.2
     hyperopt==0.2.5
     python==3.7.4


# Instructions to run the experiment
**Step 1.** Run the search process, given different random seeds.
(The wisconsin dataset is used as an example)


	(F2SAGE) python train_search.py --data wisconsin   --gpu 0  --agg sage --temp 0.001 --arch_learning_rate 0.01 --epochs 400  --learning_rate 0.02  
  
	(F2GAT)  python train_search.py --data wisconsin   --gpu 0  --agg gat  --temp 0.001 --arch_learning_rate 0.01 --epochs 400  --learning_rate 0.02    

	(F2GNN)  python train_search.py --data wisconsin   --gpu 0  --search_agg True --temp 0.001 --arch_learning_rate 0.01 --epochs 400  --learning_rate 0.02    

    (Random SAGE) python train_search.py --data wisconsin   --gpu 0  --agg sage   --algo random --temp 0.001 --arch_learning_rate 0.01 --epochs 400  --learning_rate 0.02  --alpha_mode train --random_epoch 100

The results are saved in the directory `exp_res`, e.g., `exp_res/wisconsin_sage.txt`.

**Step 2.** Fine tune the searched architectures. You need specify the arch_filename with the resulting filename from Step 1.


	python fine_tune.py --data wisconsin --gpu 0  --hyper_epoch 30  --arch_filename  exp_res/wisconsin_sage.txt --cos_lr --layer_norm True 

Step 2 is a coarse-graind tuning process, and the results are saved in a picklefile in the directory `tuned_res`, e.g., `tuned_res/wisconsin_sage.pkl`.

#### Cite
Please kindly cite [our paper](https://arxiv.org/pdf/2112.14531.pdf) if you use this code:  

    @inproceedings{wei2021designing,  
    title={Designing the Topology of Graph Neural Networks: A Novel Feature Fusion Perspective},  
    author={Wei, Lanning and Zhao, Huan and He, Zhiqiang},  
    journal={WebConf},  
    year={2022}  
    }
