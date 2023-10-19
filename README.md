## A human-like action learning process: Progressive pose generation for motion prediction
This is the code for the paper

Jinkai Li, Jinhua Wang, Ciwei Kuang, Lian Wu, Xin Wang, Yong Xu
[_"A human-like action learning process: Progressive pose generation for motion prediction"_](https://www.sciencedirect.com/science/article/abs/pii/S0950705123006986). In Knowledge-Based Systems (KBS) 2023.

### Dependencies

* cuda 11.4
* Python 3.8
* Pytorch 1.7.0

### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```shell script
h3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.
Directory structure:
```shell script
cmu_mocap
|-- test
|-- train
```

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
Put the all downloaded datasets in ./datasets directory.

### Training
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train,
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 66  --epoch 50 --num_stage 2  --l1norm 0.5
```
```bash
python main_cmu_mocap_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 16 --test_batch_size 128 --in_features 75  --epoch 50 --l1norm 0.5
```
```bash
python main_3dpw_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 69 --epoch 50  --l1norm 0.5 
```
### Evaluation
To evaluate the pretrained model,
```bash
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 66 --ckpt ./checkpoint/pretrained/main_h36m_3d/ckpt_Best_AverageError53.8396_err9.8233_err22.0336_err46.8902_err57.9075_err76.2643_err110.1189.pth
```

### Citing

If you use our code, please cite our work

```
@article{li2023human,
  title={A human-like action learning process: Progressive pose generation for motion prediction},
  author={Li, Jinkai and Wang, Jinghua and Kuang, Ciwei and Wu, Lian and Wang, Xin and Xu, Yong},
  journal={Knowledge-Based Systems},
  pages={110948},
  year={2023},
  publisher={Elsevier}
}
```

### Acknowledgments
The overall code framework is adapted from [_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://github.com/wei-mao-2019/HisRepItself) and [_Progressively Generating Better Initial Guesses Towards Next Stages for
High-Quality Human Motion Prediction_](https://github.com/705062791/PGBIG)
