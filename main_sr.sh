export OMP_NUM_THREADS=16
export MKL_NUM_THREDS=16
export NUMEXPR_NUM_THREADS=16


python3 main_sr.py --noise 0 --data 1 --set_se_const True --model gpt-4o-mini -c True
# python3 main_sr.py --noise 0 --data 2 --set_se_const True --model gpt-4o-mini
# python3 main_sr.py --noise 0 --data 3 --set_se_const True --model gpt-4o-mini
# python3 -m pdb -c continue main_gp.py --noise 0.01 --data 3