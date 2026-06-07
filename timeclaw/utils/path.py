import os.path as osp

current_dir = osp.dirname(osp.abspath(__file__))
timeclaw_dir = osp.dirname(current_dir)
root_dir = osp.dirname(timeclaw_dir)

log_dir = osp.join(root_dir, "logs")
results_dir = osp.join(root_dir, "results")

dataset_sources_dir = osp.join(root_dir, "dataset_sources")
