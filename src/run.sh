#!/bin/bash
python3 ponyge.py --parameters dnnmoo.txt | tee out-$(date +"%Y%m%d%H%M").txt

