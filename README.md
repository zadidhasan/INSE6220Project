# INSE6220Project
This is a term project for the course INSE6220 in Winter 2024. Most of the architectures can be found in the mlp.py file in the exps/baseline_h36m folder.

Here are the requirements:
1. PyTorch >= 1.5
2. Numpy
3. CUDA >= 10.1
4. Easydict
5. pickle
6. einops
7. scipy
8. six

In order to train the model, just run the following commands. It will also print the testing errors.

```
cd exps/baseline_h36m/
sh run.sh
```
After downloading the repository, copy the data folder to the following location:
```
exps/baseline_h36m/
```

