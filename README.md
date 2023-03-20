### STEPS for running SLURP offline training.

- First of all, make sure that the SLURP dataset has been downloaded. If not, you can set at lines 707-709 of the main script `download=True` (just for the first time), or you can download it from `https://zenodo.org/record/4274930`.
- The parameters in argparse are already ready for the offline training, the only thing to be changed is the `data_path`, which is the path to the repo in your system. Then, at line 41 of Speech_CLscenario/slurp_aug.py, set the path to the SLURP dataset location. 
- Create a directory where you will be storing the model after each epoch and set the name in `args.path_to_save_model`. 
- You can tweak the decoder parameters in the argparse (e.g.,`n_head_text` and `n_layer_text`) to reduce the params of the decoder.
- I suggest that you use wandb for tracking the experiments and have on-the-fly plots of the valid and train metrics, otherwise set it to False (`use_wandb`).
- To start the training stage, run:
```
python main.py --data_path MY_PATH_TO_DATASET --path_to_save_model PATH_TO_SAVE_MODEL 
```


At the end of the training, we will have the saved model after each epoch. Based on the validation metrics, you can pick the model that performed the best and use it for testing purposes. Be aware that the beam width is set to 20 by default, but it can be changed to a desired value. So, to run the test stage:

```
python main.py --data_path MY_PATH_TO_DATASET --path_to_best_model PATH_TO_BEST_MODEL 
```

The last step is to get the metrics values, using the script by the authors of the original SLURP dataset. Given that the output dictionary containing the predicted transcriptions is located at PATH_TO_PREDICTIONS, run: 
```
python evaluate.py -g <PATH_TO_GOLD> -p <PATH_TO_PREDICTIONS> --load-gold
```
