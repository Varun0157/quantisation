python -m src.quantize --q_type none 
python -m src.evaluate --q_type none

python -m src.quantize --q_type custom_whole --cpu
python -m src.evaluate --q_type custom_whole --cpu

python -m src.quantize --q_type custom_selective
python -m src.evaluate --q_type custom_selective 
