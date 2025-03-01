find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
rm -rf .ipynb_*
rm core.*
rm *.log
rm scripts/stage*_flag.txt
# rm -rf wandb