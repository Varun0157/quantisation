# quantisation
*Assignment 4* of *Advanced Natural Language Processing* (IIIT-Hyderabad, Monsoon '24)

Experiments in quantisation, consisting of quantisation from scratch (whole model and selective) as well as `bitsandbytes` integration, with quantisation to 4 bit and 8 bit formats and `nf4` quantisation. 

In addition, we deploy a model onto our local device using `llama.cpp`, quantise it, and upload it to the hugging face hub. 

## Custom Quantisation

### dependencies
Refer to the [env file](./docs/envs.yml) to install the dependencies using `conda`. 
```sh
conda env create -f docs/envs.yml
```

### quantisation
**reference**: https://github.com/ggerganov/llama.cpp/discussions/2948

Quantize `gpt-neo` using your method of choice using:
```sh
python -m src.quantize --q_type <type>
```

Types include `custom_whole`, `custom_selective`, `bnb_4`, `bnb_8`, `bnb_nf4` 
and `none`. 

`custom_whole` takes a lot of memory during inference and may have to be run with the `--cpu` flag. 

The model gets saved to `quantized`. Run it the same way you did before, instead on the evaluate model, to evaluate:
```sh
python -m src.evaluate --q_type <type>. 
```

Quantised models can be found here: https://drive.google.com/drive/folders/1lHQnaPGtltS_SNNqdw4MLhvGHB0xKP1l?usp=sharing

## llama.cpp
Set up the `llama.cpp` submodule stored in the [llama.cpp](./llama.cpp/) directory as below:
```sh
git submodule init
git submodule update
```

The remaining code assumes you're in the `llama.cpp` directory. 
```sh
cd llama.cpp
conda env create -f envs.yml && conda activate llama-cpp
```

Build the executables by referring to the [original directory](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md). 

Download `hf-smol-135m` from huggingface to quantise:
```sh
python download.py
```

Quantise the model using `llama.cpp`:
```sh
python llama.cpp/convert_hf_to_gguf.py hf-smol \
  --outfile hf-smol.gguf \
  --outtype q8_0
```

Prompt the model with whatever input you want using the `llama-cli` executable:
```sh
./llama.cpp/build/bin/llama-cli -m hf-smol.gguf -p "What is life?"
```

If you want, upload the model to hugging-face by referring to and modifying `upload.py` as required:
```sh
python upload.py
```
