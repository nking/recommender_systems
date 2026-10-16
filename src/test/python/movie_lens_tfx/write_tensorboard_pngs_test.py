import os
import shutil
import unittest
import sys
from pathlib import Path
from find_pushed_trainer_uri import get_pushed_trainer_uri

"""
USAGE from cli:

export PIPELINE_DATA_DIR=../TMP03/bin/rs_pipeline
export OUTPUT_BASE_DIR="../TMP03"
python3 -m unittest src.test.python.movie_lens_tfx.write_tensorboard_pngs_test.WriteTensorboardToPng

NOTE: can also set PUSHER_URI and TRAINER_URI, but they are automatically found if not given.
"""

is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_tfx"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_tfx"))

from helper import *
from movie_lens_tfx.utils.write_tensorboard_to_png import *

class WriteTensorboardToPng(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        pipeline_dir = os.environ.get("PIPELINE_DATA_DIR",
            os.path.join(get_bin_dir(), "rs_pipeline"))
        if not os.path.exists(pipeline_dir):
            raise ValueError(f"{pipeline_dir} does not exist")
        
        if not os.environ.get("PUSHER_URI", None):
            pusher_uri, trainer_uri = get_pushed_trainer_uri()
            cls.saved_model_dir = pusher_uri
        else:
            cls.saved_model_dir = os.environ.get("PUSHER_URI")
            trainer_uri = os.environ.get("TRAINER_URI", None)
            if not trainer_uri:
                _, trainer_uri = get_pushed_trainer_uri()
        if not os.path.exists(cls.saved_model_dir):
            raise ValueError(f"{cls.saved_model_dir} does not exist")
        if trainer_uri.find("model_run") == -1:
            trainer_uri = trainer_uri.replace("model", 'model_run')
        
        cls.log_dir = trainer_uri
        if not os.path.exists(cls.log_dir):
            raise ValueError(f"{cls.log_dir} does not exist")
        
        output_base_dir = os.environ.get("OUTPUT_BASE_DIR", get_bin_dir())

        cls.output_dir = os.path.join(output_base_dir, "pngs")
        shutil.rmtree(cls.output_dir, ignore_errors=True)
        assert(not os.path.exists(cls.output_dir))
        os.makedirs(cls.output_dir, exist_ok=True)
    
    def test_write_tensorboard_to_png(self):
        
        saved_model_dir = WriteTensorboardToPng.saved_model_dir
        logdir = WriteTensorboardToPng.log_dir
        outdir = WriteTensorboardToPng.output_dir
        
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
    
    @staticmethod
    def get_irred_error_dict(saved_model_dir: str):
        file_path = os.path.join(saved_model_dir, "assets.extra",
            "irreducible_error.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                irr_err_dict = json.load(f)
                return irr_err_dict
        return None
    
    @staticmethod
    def get_hyperparams_dict(saved_model_dir: str):
        file_path = os.path.join(saved_model_dir, "assets.extra",
            "hyperparameters.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                h_dict = json.load(f)
                return h_dict['values']
        return None
    
    @staticmethod
    def calc_random_inbatch_softmax_loss(batch_size:int):
        return np.log(batch_size)
    
    @staticmethod
    def calc_random_ndcg(k: int, movie_catalog_size: int) -> float:
        s = np.sum([1 / np.log2(r + 1) for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()
    
    @staticmethod
    def calc_random_recall(k: int, movie_catalog_size: int) -> float:
        return k / movie_catalog_size
    
    @staticmethod
    def calc_random_precision(k: int, movie_catalog_size: int) -> float:
        return 1 / movie_catalog_size
    
    @staticmethod
    def calc_random_mrr(k: int, movie_catalog_size: int) -> float:
        #assuming 1 relevant item
        s = np.sum([1/r for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()
    
    def find_irred_dict(self, irred_err_dict, metric_name):
        #irred_err_dict keys: {'composite_ndcg_k', 'hit_rate', 'mean_loss', 'mrr_k',
        # 'ndcg_k', 'ndcg_head_k', 'ndcg_tail_k','ndcg_torso_k', 'recall_k'}
        # metric_name usually starts with epoch_
        if irred_err_dict is None:
            return None
        metric_name = metric_name.replace("epoch_", "")
        return irred_err_dict.get(metric_name, None)
    
if __name__ == '__main__':
    unittest.main()
