# Supplementary code for "Just a Simple Transformation is Enough for Data Protection in Vertical Federated Learning"
## How to install
```bash
git clone https://github.com/Anonymousedauthor777/simple-tranformations
cd simple-tranformations
pip install -r requirements.txt
```
## Repository structure
* <ins>unsplit</ins> contains examples of defense against the model inversion attack on MLPs, and differentially private protection for both CNN and MLP architectures.
* In <ins>fsha</ins> , you may find the code for Hijacking attack on models with and without the dense layer and make sure of the data protection.
* Please, run  ```FSHA.ipynb```, ```model_inversion_stealing.ipynb```, ```dp_defense.ipynb``` to observe the defense results at your convenience.
## Implementation details
Our work defends against the model inversion attack from ["Unsplit" (Erdogan et al., 2022)](https://arxiv.org/abs/2108.09033) ([code](https://github.com/ege-erdogan/unsplit))and Feature-Space-Hijacking attack (FSHA) from ["Unleashing the Tiger" (Pasquini et al., 2021)](https://arxiv.org/abs/2012.02670) ([code](https://github.com/pasquini-dario/SplitNN_FSHA)).

In both cases, the necessary hyperparameters are required, we list them below:

- Common arguments for Split Learning protocol are: ```batch_size```, ```split_layer```, ```dataset_name```, ```device```, ```n_epochs```, ```architecture```.
- We conduct all experiments on ```mnist``` and ```f_mnist``` datasets, for this purposes assign the proper name to ```dataset_name```. And the main hyperparameter for validating our results is ```split_layer```, feel free to set its number from ```1``` to ```6```. 
- Set the ```architecture``` to either ```mlp``` or ```cnn```. In case of ```cnn``` you will see the original performance of *Unsplit* and *FSHA* (except of the DP setup).
Below, we describe the changes for the two mentioned settings.

- <ins>fsha</ins> folder:
    - For *FSHA* we use some special hyperparameters: ```WGAN```, ```gradient_penalty```, ```style_loss```, ```lr_f```, ```lr_tilde```, ```lr_D```. These hyperparameters refer to the training of the encoder, decoder, and discriminator networks; we took them from the original implementation (see [code](https://github.com/pasquini-dario/SplitNN_FSHA/blob/main/FSHA.ipynb)) and did not change them in our work.
    - The changes occur in ```architectures.py```, where we introduce ```pilot_mlp```, ```discriminator_mlp``` and left the same ```pilot_cnn```  with ```discriminator_cnn```.  Set the ```architecture``` value to ```mlp``` to observe the *FSHA* on mlp-based model, while the core architecture in ```cnn``` case is ```resnet```.

- <ins>unsplit</ins> folder:
    - For *Unsplit* we mention other special hyperparameters: ```lr```, ```main_iters```,```input_iters```, ```model_iters```, ```lambda_l2```, ```lambda_tv```. We suggest configuring them as laid out in our work (```0.001```, ```200```, ```20```, ```20```, ```0.1```, ```1.```) for efficient reproduction of the results. We also stress that ```lambda_l2``` regularizer was not mentioned in the original *Unsplit* paper's model inversion attack algorithm.
    - When it comes to the DP setting, we use the same training hyperparameters as those used in defense with MLPs against *Unsplit*. The difference lies in the code for adding noise to the dataloader. The key hyperparameters, in this case, are: ```epsilon``` and ```delta``` for the global $\ell_2$ sensitivity. We use ```calibrateAnalyticGaussianMechanism``` from [Borja Balle et al., 2018](https://arxiv.org/abs/1805.06530) [code](https://github.com/BorjaBalle/analytic-gaussian-mechanism/blob/master/agm-example.py) to calculate ```sigma``` for each of the mentioned datasets. For achieving a proper utility-privacy trade-off, we suggest picking ```epsilon=6```, ```delta=0.5```, ```n_epochs=20``` for ```mnist``` and ```f_mnist``` (so the value of $\sigma$ equals to ```1.6``` and ```2.6```, respectively).
    - We also conducted an experiments on the CIFAR10 dataset, which we report in Table 3.  For these experiments, please refer to the ```additional_dp_experiments.ipynb```. We used ```n_epochs=50``` and ```epsilon```, ```delta``` that result in ```sigma=0.25``` for CIFAR10.

**We believe the details provided are clear enough to reproduce the main findings of our paper.**