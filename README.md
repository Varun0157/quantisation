# quantisation
Assignment 4 of Advanced Natural Language Processing (Monsoon '24)

___
build the dependencies by referring to the env files in docs. 

Retrieve the models by running `python scripts/get_models.py`. 

Quantize `gpt-neo` using your method of choice using `python -m src.quantize --q_type <type>`. 
Types include `custom_whole`, `custom_selective`, `bnb_4`, `bnb_8`, `bnb_nf4` and `none`. 
`custom_whole` takes a lot of memory during inference and may have to be run with the `--cpu` flag. 

The model gets saved to `quantized`. Run it the same way you did before, instead on the evaluate model, to evaluate:
`python -m src.evaluate --q_type <type>`. 

Trained models can be found here: https://drive.google.com/drive/folders/1lHQnaPGtltS_SNNqdw4MLhvGHB0xKP1l?usp=sharing
