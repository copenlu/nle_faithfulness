

## NLE Model Training
The implementation is based on the implementation of the paper "Leakage-Adjusted Simulatability: Can Models Generate Non-Trivial Explanations of Their Behavior in Natural Language?", https://github.com/peterbhase/LAS-NL-Explanations

We train NLE generation models in four different setups using the sim_experiments/run_tasks.py script.
Training of the models takes the following form (example for e-SNLI):
```
#Task Model:
python run_tasks.py --gpu 0 -e NLI.task -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache --server_number 1
#Human Simulator:
python run_tasks.py --gpu 0 -e NLI.SIM.human -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache --server_number 1

#MT-Re: 
python run_tasks.py --gpu 0 -e NLI.CLM.reason.MT -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache --server_number 1
#Simulator:
python run_tasks.py --gpu 0 -e NLI.SIM.MT.RE -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache --server_number 1

#*ST-Re*: 
#Generator:
python run_tasks.py --gpu 0 -e NLI.CLM.reason -b 6 -g 6 --save_dir models/general/nli --cache_dir nli_cache --server_number 1
#Task model:
python run_tasks.py --gpu 0 -e NLI.ST.RE -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache --server_number 1
#Simulator:
python run_tasks.py --gpu 0 -e NLI.SIM.ST.RE -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache --server_number 1

#MT-Ra: 
#Task model:
python run_tasks.py --gpu 0 -e NLI.CLM.rationalize.MT -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache  --server_number 1
#Simulator:
python run_tasks.py --gpu 0 -e NLI.SIM.MT.RA -b 4 -g 3 --save_dir models/general/nli --cache_dir nli_cache  --server_number 1

#ST-Ra: 
#Generator:
python run_tasks.py --gpu 0 -e NLI.CLM.rationalize -b 6 -g 6 --save_dir models/general/nli --cache_dir cache  --server_number 1
#Task model:
python run_tasks.py --gpu 0 -e NLI.ST.RA -b 4 -g 3 --save_dir models/general/nli --cache_dir cache --server_number 1
#Simulator:
python run_tasks.py --gpu 0 -e NLI.SIM.ST.RA -b 4 -g 3 --save_dir models/general/nli --cache_dir cache --server_number 1
```
For ComVE, and CoS-E tasks, the name of the task NLI should be replaced with COMVE and QA, correspondingly.

## Counterfactual Test

### Random Baseline

The random baseline a random adjective before a noun or a random adverb before a verb and evaluates whether the inserted word 1) changes the prediction of the model and 2) is found in the generated explanation for the instance.
The insertion procedure takes the following form (example for e-SNLI):

```
python3 counterfactual/random_baseline.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 8
python3 counterfactual/random_baseline.py --task_pretrained_name t5-base  --model_name MT.RE --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RE_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RE_seed21 --explanations_to_use t5-MT-single-exp-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 8
python3 counterfactual/random_baseline.py --task_pretrained_name t5-base --model_name ST.RE --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name ST.RE_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_ST.RE_seed21 --explanations_to_use t5-single-exp-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 16
python3 counterfactual/random_baseline.py --task_pretrained_name t5-base --multi_explanation true --model_name ST.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name ST.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_ST.RA_seed21 --explanations_to_use t5-multi-exp-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 16 --condition_on_explanations True
```

### Counterfactual Editor

The counterfactual editor employs a model trained to produce input inserts that change the prediction of the model.
The training of the editor takes the following form (example for e-SNLI):

```
python3 counterfactual/counterfactual_editor.py --task_pretrained_name t5-base --multi_explanation true --model_name MT.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RA_seed21 --explanations_to_use t5-MT-multi-exp-pred-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 16 --num_train_epochs 3 --max_words_gen 3 --lr 4e-5 --num_beams 4 --train_batch_size 16 
python3 counterfactual/counterfactual_editor.py --task_pretrained_name t5-base --model_name MT.RE --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name MT.RE_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_MT.RE_seed21 --explanations_to_use t5-MT-single-exp-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 16 --num_train_epochs 3 --max_words_gen 3 --lr 4e-5 --num_beams 4 --train_batch_size 16 
python3 counterfactual/counterfactual_editor.py --task_pretrained_name t5-base --model_name ST.RE --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name ST.RE_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_ST.RE_seed21 --explanations_to_use t5-single-exp-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 16 --num_train_epochs 3 --max_words_gen 3 --lr 4e-5 --num_beams 4 --train_batch_size 32 
python3 counterfactual/counterfactual_editor.py --task_pretrained_name t5-base --multi_explanation true --model_name ST.RA --save_dir models/general/nli/  --cache_dir nli_cache --data_dir data/e-SNLI-data --prefinetuned_name ST.RA_seed21 --seed 21 --labels_to_use preds_NLI_t5-base_ST.RA_seed21 --explanations_to_use t5-multi-exp-seed21 --n_pos 4 --num_return_sequences 4 --dev_batch_size 16 --condition_on_explanations True --num_train_epochs 3 --max_words_gen 3 --lr 4e-5 --num_beams 4 --train_batch_size 32
```

## Input Reconstruction Test
For the input reconstruction test, one has to 1) create the reconstructed text input 2) run evaluation on the new dataset with run_tasks.py dataset and 3) compare the outputs on the new dataset. The reconstructed text for e-SNLI leverages the template form present in many of the generated NLEs. The procedure takes the form as in sim_expriments/Templates-SNLI.ipynb (for e-SNLI dataset).