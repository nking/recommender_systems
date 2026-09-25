import os
import shutil
import unittest

from helper import *
from movie_lens_tfx.utils.write_tensorboard_to_png import *

class WriteTensorboardToPng(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def get_irred_error_dict(self, saved_model_dir:str):
        file_path = os.path.join(saved_model_dir, "assets.extra", "irreducible_error.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                irr_err_dict = json.load(f)
                return irr_err_dict
    
    def test_write_tensorboard_to_png(self):
        outdir = os.path.join(get_bin_dir(), "pngs")
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)
        
        p = "rs_pipeline/Trainer/model_run/19"
        logdir = os.path.join(get_bin_dir(), p)
        saved_model_dir = os.path.join(get_bin_dir(), "rs_pipeline/Pusher/pushed_model/21")
        
        #temporarily point to recently trained model:
        #logdir = os.path.join(get_project_dir(), "../TMP8/bin", p)
        #saved_model_dir = os.path.join(get_project_dir(), "../TMP8/bin/rs_pipeline/Pusher/pushed_model/21")
        
        #logdir = os.path.join(get_bin_dir(), "TestPipelines_baseline/MAIN_USER_MOVIE/Trainer/model_run/17/")
        #saved_model_dir = os.path.join(get_bin_dir(), "TestPipelines_baseline/MAIN_USER_MOVIE/Pusher/pushed_model/19")
        
        train_dir = os.path.join(logdir, "train")
        val_dir = os.path.join(logdir, "validation")
        test_dir = os.path.join(logdir, "test")
        irred_err_dict = self.get_irred_error_dict(saved_model_dir)
        metrics = list_tfevents_metrics(train_dir)
        print(f'metrics: {metrics}', flush=True)
        
        for metric in metrics:
            outfile = os.path.join(outdir, f"{metric}.png")
            irr_dict = self.find_irred_dict(irred_err_dict, metric)
            if metric.find("recall") > -1:
                irr_dict = None
            generate_tensorboard_chart(train_dir, val_dir, test_dir, irr_dict,
                scalar_name=metric, output_path=outfile)
        
        for metric in ['epoch_pre_logit_min', 'epoch_pre_logit_mean', 'epoch_pre_logit_max',
        'epoch_learning_rate']:
            outfile = os.path.join(outdir, f"{metric}.png")
            export_scalars_to_png(train_dir, outfile, metric)
            
        print(f'wrote pngs to {outdir}')
    
    def find_irred_dict(self, irred_err_dict, metric_name):
        #irred_err_dict keys: {'composite_ndcg_20', 'hit_rate', 'mean_loss', 'mrr_20',
        # 'ndcg_20', 'ndcg_head_20', 'ndcg_tail_20','ndcg_torso_20', 'recall_20'}
        # metric_name usually starts with epoch_
        metric_name = metric_name.replace("epoch_", "")
        return irred_err_dict.get(metric_name, None)
        