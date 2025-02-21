find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
rm -rf .ipynb_*
rm core.*
rm -rf /ssdwork/liuyang/Checkpoint/SSR*
rm *.log
rm scripts/*flag.txt
rm -rf wandb