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
            
    def get_hyperparams_dict(self, saved_model_dir:str):
        file_path = os.path.join(saved_model_dir, "assets.extra", "hyperparameters.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                h_dict = json.load(f)
                return h_dict['values']
    
    def test_write_tensorboard_to_png(self):
        outdir = os.path.join(get_bin_dir(), "pngs")
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)
        
        p = "rs_pipeline/Trainer/model_run/19"
        logdir = os.path.join(get_bin_dir(), p)
        saved_model_dir = os.path.join(get_bin_dir(), "rs_pipeline/Pusher/pushed_model/21")
        
        #temporarily point to recently trained model:
        logdir = os.path.join(get_project_dir(), "../TMP10/bin", p)
        saved_model_dir = os.path.join(get_project_dir(), "../TMP10/bin/rs_pipeline/Pusher/pushed_model/21")
        
        #logdir = os.path.join(get_bin_dir(), "TestPipelines_baseline/MAIN_USER_MOVIE/Trainer/model_run/17/")
        #saved_model_dir = os.path.join(get_bin_dir(), "TestPipelines_baseline/MAIN_USER_MOVIE/Pusher/pushed_model/19")
        
        hyperparams_dict = self.get_hyperparams_dict(saved_model_dir)
        random_ndcg = self.calc_random_ndcg(k=hyperparams_dict['k'], movie_catalog_size=hyperparams_dict['n_movies'])
        random_recall = self.calc_random_recall(k=hyperparams_dict['k'], movie_catalog_size=hyperparams_dict['n_movies'])
        random_precision = self.calc_random_precision(k=hyperparams_dict['k'],
            movie_catalog_size=hyperparams_dict['n_movies'])
        random_mrr = self.calc_random_mrr(k=hyperparams_dict['k'],
            movie_catalog_size=hyperparams_dict['n_movies'])
        random_loss = self.calc_random_inbatch_softmax_loss(hyperparams_dict['BATCH_SIZE'],)

        train_dir = os.path.join(logdir, "train")
        val_dir = os.path.join(logdir, "validation")
        test_dir = os.path.join(logdir, "test")
        irred_err_dict = self.get_irred_error_dict(saved_model_dir)
        metrics = list_tfevents_metrics(train_dir)
        print(f'metrics: {metrics}', flush=True)
        
        for metric in metrics:
            outfile = os.path.join(outdir, f"{metric}.png")
            random_metric = None
            irr_dict = self.find_irred_dict(irred_err_dict, metric)
            if metric.find("recall") > -1:
                irr_dict = None
                random_metric = random_recall
            elif metric.find("ndcg") > -1:
                random_metric = random_ndcg
            elif metric.find("hit_rate") > -1:
                #for 1 relevant ground_truth item:
                random_metric = random_ndcg
            elif metric.find("mrr") > -1:
                #for 1 relevant ground_truth item:
                random_metric = random_mrr
            elif metric.find("precision") > -1:
                #for 1 relevant ground_truth item:
                random_metric = random_precision
            elif metric.find("loss") > -1:
                random_metric = random_loss
            generate_tensorboard_chart(train_dir, val_dir, test_dir, irr_dict, random_metric,
                scalar_name=metric, output_path=outfile)
        
        for metric in ['epoch_pre_logit_min', 'epoch_pre_logit_mean', 'epoch_pre_logit_max',
        'epoch_learning_rate']:
            outfile = os.path.join(outdir, f"{metric}.png")
            export_scalars_to_png(train_dir, outfile, metric)
            
        print(f'wrote pngs to {outdir}')
    
    def calc_random_inbatch_softmax_loss(self, batch_size:int):
        return np.log(batch_size)
    
    def calc_random_ndcg(self, k: int, movie_catalog_size: int) -> float:
        s = np.sum([1 / np.log2(r + 1) for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()
    
    def calc_random_recall(self, k: int, movie_catalog_size: int) -> float:
        return k / movie_catalog_size
    
    def calc_random_precision(self, k: int, movie_catalog_size: int) -> float:
        return 1 / movie_catalog_size
    
    def calc_random_mrr(self, k: int, movie_catalog_size: int) -> float:
        #assuming 1 relevant item
        s = np.sum([1/r for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()
    
    def find_irred_dict(self, irred_err_dict, metric_name):
        #irred_err_dict keys: {'composite_ndcg_k', 'hit_rate', 'mean_loss', 'mrr_k',
        # 'ndcg_k', 'ndcg_head_k', 'ndcg_tail_k','ndcg_torso_k', 'recall_k'}
        # metric_name usually starts with epoch_
        metric_name = metric_name.replace("epoch_", "")
        return irred_err_dict.get(metric_name, None)
        