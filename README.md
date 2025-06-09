# DL-Adnexal-Lesions

## Checkpoint
The checkpoint of our OCNet model can be obtained here: https://drive.google.com/drive/folders/1j9Z0JnMU8M0taOezLDdNwhREF411IMog?usp=drive_link

## Requirement
We provide the necessary toolkit dependency file requirements.txt

## Seg
Regarding our segmentation model, we provide it in the seg folder.

## Train
In the project path, run python train.py. Our hyperparameters have default settings in main.py. If you need to change them, just modify them in the command line.
```shell
python train.py
```

## Eval
To verify our model, we only need to load the corresponding checkpoint and then run python evaluate2.py. The specific parameter settings also use the best results by default.
```shell
python evaluate2.py
```

## seg
If you want to train or verify the effectiveness of the segmentation model, enter the seg directory and run the corresponding train.py and test.py. Our best parameters are also placed in the code by default.





