# quantisation
*Assignment 4* of *Advanced Natural Language Processing* (IIIT-Hyderabad, Monsoon '24)

Experiments in quantisation, consisting of quantisation from scratch (whole model and selective) as well as `bitsandbytes` integration, with quantisation to 4 bit and 8 bit formats and `nf4` quantisation. 

In addition, we deploy a device onto our local device using `llama.cpp`, quantise it, and upload it to the hugging face hub. 

## Custom Quantisation
___
build the dependencies by referring to the env files in docs. 

Quantize `gpt-neo` using your method of choice using `python -m src.quantize --q_type <type>`. 
Types include `custom_whole`, `custom_selective`, `bnb_4`, `bnb_8`, `bnb_nf4` and `none`. 
`custom_whole` takes a lot of memory during inference and may have to be run with the `--cpu` flag. 

The model gets saved to `quantized`. Run it the same way you did before, instead on the evaluate model, to evaluate:
`python -m src.evaluate --q_type <type>`. 

Trained models can be found here: https://drive.google.com/drive/folders/1lHQnaPGtltS_SNNqdw4MLhvGHB0xKP1l?usp=sharing

___

For the gguf files in the submission, some reference commands ran were attached. These have to be run in a directory that has llama.cpp cloned and built. 
