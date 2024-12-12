python -m src.quantize --q_type none 
python -m src.evaluate --q_type none

python -m src.quantize --q_type custom_whole --cpu
python -m src.evaluate --q_type custom_whole --cpu

python -m src.quantize --q_type custom_selective
python -m src.evaluate --q_type custom_selective 

python -m src.quantize --q_type bnb_4 
python -m src.evaluate --q_type bnb_4

python -m src.quantize --q_type bnb_8 --cpu
python -m src.evaluate --q_type bnb_8 --cpu

python -m src.quantize --q_type bnb_4_nf4
python -m src.evaluate --q_type bnb_4_nf4
