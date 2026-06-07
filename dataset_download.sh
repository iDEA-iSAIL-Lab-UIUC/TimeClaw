#!/bin/bash
# Download benchmark datasets into ./dataset_sources/.
#
# After running this script the layout should be:
#   dataset_sources/
#     CiK/all_tasks.json                  # from the Context-is-Key release
#     TSAIA/                              # HuggingFace dataset
#     TSRBench/{perception,prediction,decision,reasoning}/*.jsonl   # HuggingFace dataset
#
# CiK is NOT on HuggingFace; download `all_tasks.json` from the official
# Context-is-Key benchmark release and place it at
# `dataset_sources/CiK/all_tasks.json` before running the CiK evaluator.

set -e
mkdir -p dataset_sources
cd dataset_sources

git clone https://huggingface.co/datasets/Melady/TSAIA
git clone https://huggingface.co/datasets/umd-zhou-lab/TSRBench

mkdir -p CiK
echo "Place all_tasks.json under dataset_sources/CiK/ to enable the CiK evaluator."

cd ..
