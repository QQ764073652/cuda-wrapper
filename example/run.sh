#!/bin/bash
echo "Virtual GPU Memory Size Quota:${VOLCANO_GPU_ALLOCATED}MB"

echo "Batch Size = 10"
python /example/code.py 10

echo "Batch Size = 100"
python /example/code.py 100

echo "Batch Size = 1000"
python /example/code.py 1000

echo "Batch Size = 2000"
python /example/code.py 2000 || exit 0


