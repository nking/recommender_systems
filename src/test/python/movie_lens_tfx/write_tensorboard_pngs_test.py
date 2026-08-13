import shutil
import unittest

from helper import *
from movie_lens_tfx.utils.write_tensorboard_to_png import *

class WriteTensorboardToPng(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_write_tensorboard_to_png(self):
        outdir = os.path.join(get_bin_dir(), "pngs")
        try:
            shutil.rmtree(outdir)
        except OSError:
            pass
        os.makedirs(outdir, exist_ok=True)
        
        p = "rs_pipeline/Trainer/model_run/19"
        logdir = os.path.join(get_bin_dir(), p)
        train_dir = os.path.join(logdir, "train")
        val_dir = os.path.join(logdir, "validation")
        metrics = list_tfevents_metrics(train_dir)
        
        for metric in metrics:
            outfile = os.path.join(outdir, f"{metric}.png")
            generate_tensorboard_chart(train_dir, val_dir, scalar_name=metric, output_path=outfile)
        print(f'wrote pngs to {outdir}')