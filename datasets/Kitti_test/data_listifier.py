#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path
import argparse
import numpy as np
import glob

parser = argparse.ArgumentParser(description = 'read filenames from the give directory and turn into list in .txt file for dataloader')
parser.add_argument('--r', required=True, help='designate root directory')
parser.add_argument('--out', required=False, help='output .txt file name')
parser.add_argument('--no_ext', required=False, default=False, action='store_true', help='record without extension')
args = parser.parse_args()

root_path = args.r
if not root_path.startswith('.'):
    root_path = './' + root_path

output_filename = 'data_list.txt'
if args.out != None:
    output_filename = args.out

with open(output_filename, 'w') as f:
    #file_list_npy = [file for file in os.listdir(root_path) if file.endswith('npy')]
    file_list_npy = sorted(glob.glob(root_path + '/*.npy', recursive=False))
    for filename in file_list_npy:
        if args.no_ext:
            f.write(filename.split('.npy')[0] + '\n')
        else:
            f.write(filename + '\n')