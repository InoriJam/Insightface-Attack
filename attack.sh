#!/bin/bash
source activate tensorflow
export PYTHONPATH="./src:$PYTHONPATH"
python ./attack.py --data ./lfw/new112 \
	--eps 25
#python L2_calc.py
zip -r images.zip ./images > /dev/null

