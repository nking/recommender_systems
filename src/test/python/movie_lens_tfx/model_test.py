
import numpy as np
import tensorflow as tf
import os

from helper import *
from movie_lens_tfx.tune_train_movie_lens import _make_2tower_keras_model, \
    get_default_hyperparameters


class ModelTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
    
    def test_tier_weighted_in_batch_softmax_loss(self):
        custom_config :  dict[str, Any]= {
          'n_users': 6040,
          'n_movies': 3883,
          'n_genres': 18,
          'feature_acronym': "ahosy",
          'run_eagerly': False,
          'incl_genres': True,
          'BATCH_SIZE':512,
          "NUM_EPOCHS":10,
          "num_train_examples" : 459586,
          "num_val_examples" : 459586,
          "num_test_examples" : 459586,
          "version" : 1.0,
          "model_name" : "test_model",
          "movie_tiers_uri" : os.path.join(get_project_dir(), "src/test/resources/movie_tiers.json"),
          "input_dataset_element_spec_trans_ser" : 'gASV/gIAAAAAAAB9lCi'
              'MA2FnZZSMInRlbnNvcmZsb3cucHl0aG9uLmZyYW1ld29yay50ZW5zb3KU'
               'jApUZW5zb3JTcGVjlJOUjCh0ZW5zb3JmbG93LnB5dGhvbi5mcmFtZXdvcms'
                'udGVuc29yX3NoYXBllIwLVGVuc29yU2hhcGWUk5RdlChoBYwJRGltZW5zaW'
                '9ulJOUToWUUpRoCksBhZRSlGWFlFKUjCJ0ZW5zb3JmbG93LnB5dGhvbi5mcmFtZXdvcmsuZHR5cGVzlIwIYXNfZHR5cGWUk5SM'
                 'B2Zsb2F0MzKUhZRSlGgBh5RSlIwGZ2VuZGVylGgEaAddlChoCk6FlFKUaApLAYWUUpRlhZRSlGgWaBmHlFKUjAZnZW5yZXOUa'
              'ARoB12UKGgKToWUUpRoCksBhZRSlGgKSxKFlFKUZYWUUpRoFmgjh5RSlIwCaHKUaARoB12UKGgKToWUUpRoCksBhZRSlGWFlFKUaBZo'
               'L4eUUpSMBWhyX3drlGgEaAddlChoCk6FlFKUaApLAYWUUpRlhZRSlGgWaDmHlFKUjAVtb250aJRoBGgHXZQoaApOhZRSlGgKSwGFlFKU'
                'ZYWUUpRoFmhDh5RSlIwIbW92aWVfaWSUaARoB12UKGgKToWUUpRoCksBhZRSlGWFlFKUaBZoTYeUUpSMCm9jY3VwYXRpb26UaARoB12'
               'UKGgKToWUUpRoCksBhZRSlGWFlFKUaBZoV4eUUpSMC3NlY19pbnRvX3lylGgEaAddlChoCk6FlFKUaApLAYWUUpRlhZRSlGgWaGGHlFK'
                'UjAd1c2VyX2lklGgEaAddlChoCk6FlFKUaApLAYWUUpRlhZRSlGgWaGuHlFKUjAd3ZWVrZGF5lGgEaAddlChoCk6FlFKUaApLAYWUUpR'
               'lhZRSlGgWaHWHlFKUjAJ5cpRoBGgHXZQoaApOhZRSlGgKSwGFlFKUZYWUUpRoFmh/h5RSlIwEeXJfepRoBGgHXZQoaApOhZRSlGgKSwGFlFKUZYWUUpRoFmiJh5RSlHUu'
        }
        hp = get_default_hyperparameters(custom_config)
        hp.Fixed('input_dataset_element_spec_trans_ser', value=custom_config["input_dataset_element_spec_trans_ser"])
        model = _make_2tower_keras_model(hp)
        self.run_tier_weighted_in_batch_softmax_loss(model)
    
    def run_tier_weighted_in_batch_softmax_loss(self, model):
        B = 512
        # Diagonal logits = 2.0, off-diagonal = 0.0 -> identical per-row CE loss across all rows
        logits = tf.eye(B, dtype=tf.float32) * 2.0
        y_ones = tf.ones((B, 1), dtype=tf.float32)
    
        # Skewed batch: 333 Head (65%), 128 Torso (25%), 51 Tail (10%)
        tiers_skewed = tf.constant([0] * 333 + [1] * 128 + [2] * 51, dtype=tf.int32)
    
        #Scale Invariance: When every row has identical logits and error,
        # the tier-weighted loss equals the unweighted loss to within 1e-6 (4.250726 == 4.250726).
        loss_unweighted = model.in_batch_softmax_loss_function(
            y_true=y_ones, logits=logits, batch_movie_tiers=None
        )
        loss_tier_weighted = model.in_batch_softmax_loss_function(
            y_true=y_ones, logits=logits, batch_movie_tiers=tiers_skewed
        )
        tf.debugging.assert_near(loss_unweighted, loss_tier_weighted, atol=1e-5)
    
        # Exact 1/3 Gradient Budget Share per Tier via GradientTape
        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss_val = model.in_batch_softmax_loss_function(
                y_true=y_ones, logits=logits, batch_movie_tiers=tiers_skewed
            )
        grads = tape.gradient(loss_val, logits)  # [B, B]
        row_grad_norms = tf.reduce_sum(tf.abs(grads), axis=1)  # [B]
        total_grad_norm = tf.reduce_sum(row_grad_norms)
    
        for tier_idx in [0, 1, 2]:
            tier_share = (
                tf.reduce_sum(tf.boolean_mask(row_grad_norms, tiers_skewed == tier_idx))
                / total_grad_norm
            )
            tf.debugging.assert_near(tier_share, 1.0 / 3.0, atol=1e-4)
    
        # Missing-Tier Renormalization Test (0 Tail items -> 50% Head, 50% Torso)
        tiers_no_tail = tf.constant([0] * 384 + [1] * 128, dtype=tf.int32)
        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss_no_tail = model.in_batch_softmax_loss_function(
                y_true=y_ones, logits=logits, batch_movie_tiers=tiers_no_tail
            )
        grads_no_tail = tape.gradient(loss_no_tail, logits)
        row_norms_no_tail = tf.reduce_sum(tf.abs(grads_no_tail), axis=1)
        head_share = tf.reduce_sum(row_norms_no_tail[:384]) / tf.reduce_sum(row_norms_no_tail)
        torso_share = tf.reduce_sum(row_norms_no_tail[384:]) / tf.reduce_sum(row_norms_no_tail)
    
        tf.debugging.assert_near(loss_no_tail, loss_unweighted, atol=1e-5)
        tf.debugging.assert_near(head_share, 0.50, atol=1e-4)
        tf.debugging.assert_near(torso_share, 0.50, atol=1e-4)
    
        # Single-Outlier 5% Denominator Floor Cap Test (1 Tail item in B=512)
        tiers_one_tail = tf.constant([0] * 383 + [1] * 128 + [2] * 1, dtype=tf.int32)
        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss_one_tail = model.in_batch_softmax_loss_function(
                y_true=y_ones, logits=logits, batch_movie_tiers=tiers_one_tail
            )
        grads_one_tail = tape.gradient(loss_one_tail, logits)
        row_norms_one_tail = tf.reduce_sum(tf.abs(grads_one_tail), axis=1)
    
        # Compare the single tail item's gradient norm against an unweighted uniform row norm
        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss_unif = model.in_batch_softmax_loss_function(
                y_true=y_ones, logits=logits, batch_movie_tiers=None
            )
        grads_unif = tape.gradient(loss_unif, logits)
        unif_row_norm = tf.reduce_sum(tf.abs(grads_unif[0]))
    
        outlier_boost = row_norms_one_tail[-1] / unif_row_norm
        # Must be capped at (0.33 / 0.99) / 0.05 = 6.6667x instead of 512 / 3 = 170.67x
        tf.debugging.assert_near(outlier_boost, (1.0 / 3.0) / 0.05, atol=1e-3)
    
        # @tf.function Dynamic Batch Shape Trace Test
        #Confirms XLA/Autograph traces dynamic [None, None] batch sizes without shape-broadcasting bugs
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
            ]
        )
        def traced_loss(y_in, logits_in, tiers_in):
            return model.in_batch_softmax_loss_function(y_in, logits_in, tiers_in)
    
        traced_out = traced_loss(y_ones[:17], logits[:17, :17], tiers_skewed[:17])
        tf.debugging.assert_all_finite(traced_out, "Traced loss produced NaN/Inf")
        print("All 5 in_batch_softmax_loss_function validation tests passed!")