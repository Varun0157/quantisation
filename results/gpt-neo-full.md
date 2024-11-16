❯ python -m src.main --q_type custom
INFO - 2024-11-16 22:38:22,752 : Loading PennTreeBank dataset...
INFO - 2024-11-16 22:38:26,924 : PennTreeBank dataset loaded
INFO - 2024-11-16 22:38:27,796 : device: cpu
INFO - 2024-11-16 22:38:29,336 : running pre-trained model ... 
calculating perplexity...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [04:48<00:00, 10.38it/s]
INFO - 2024-11-16 22:43:18,315 : perplexity: 48.26491300833854
INFO - 2024-11-16 22:43:18,315 : average latency: 0.09573393893241883
INFO - 2024-11-16 22:43:18,315 : memory footprint: 551.126064 MB
INFO - 2024-11-16 22:43:18,315 : quantizing all layers of model to torch.int8 (linear) ...
INFO - 2024-11-16 22:43:20,193 : running quantized model ... 
calculating perplexity...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [13:26<00:00,  3.72it/s]
INFO - 2024-11-16 22:56:46,581 : perplexity: 48.137496477235786
INFO - 2024-11-16 22:56:46,581 : average latency: 0.26814657672246295
INFO - 2024-11-16 22:56:46,581 : memory footprint: 215.093276 MB