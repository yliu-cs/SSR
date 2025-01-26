find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
rm -rf .ipynb_*
rm core.*
rm -rf checkpoint
rm *.log