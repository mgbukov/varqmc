#!/bin/sh
 
data_dir="$(python make_data_file.py)"
echo $data_dir

python main.py ${data_dir}