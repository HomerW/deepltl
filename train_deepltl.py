import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping1D, Input, Reshape
import numpy as np
import pickle
from LTLOperator import LTLOperator
import os
import argparse
from models import get_model_zero, get_model_one, get_model_two
from train_utils import DiscreteAcc, Anneal

batch_size = 100
num_restarts = 2
num_formulas = 50

"""
Trains the DeepLTL models on data from data_path and
saves checkpoints to train_path.
"""
def train_models(models, data_path, train_path):
    for name, get_model in enumerate(models):
        for n in range(1, 16):
            count = 0
            # some formulas may be skipped so read in as many files
            # as it takes to get to num_formulas (up to 100)
            for i in range(100):
                if count >= num_formulas:
                    break
                checkpoint_path = f"{train_path}/{name}/{n}/cp-{i}.ckpt"

                if os.path.exists(f"{data_path}/{n}/train-{i}"):
                    with open(f"{data_path}/{n}/train-{i}", "rb") as file:
                      train_traces, train_labels = pickle.loads(file.read())

                    best_acc = 0
                    for restart in range(num_restarts):
                        model = get_model()
                        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
                                      loss='binary_crossentropy',
                                      metrics=['accuracy'])

                        history = model.fit(train_traces, train_labels,
                                            epochs=3000, batch_size=batch_size,
                                            verbose=0,
                                            shuffle=True,
                                            callbacks=[Anneal(),
                                                       DiscreteAcc(train_traces, train_labels)])
                        output = model.predict(train_traces, batch_size=batch_size)
                        labels = train_labels.numpy()
                        acc = float(tf.reduce_mean(tf.cast(output.reshape(-1) == labels, tf.float32)))
                        if acc >= best_acc:
                            model.save_weights(checkpoint_path)
                            best_acc = acc
                        if best_acc == 1.0:
                            break
                    count += 1

                    print(f"Trained formula size {n} number {count} with accuracy {best_acc}")

                tf.keras.backend.clear_session()

if __name__ == "__main__":
    models = [get_model_zero, get_model_one, get_model_two]
    parser = argparse.ArgumentParser(description="Inputs for DeepLTL training")
    parser.add_argument('--data_path',required=True,type=str,help="Path to traces")
    parser.add_argument('--train_path',required=True,type=str,help="Path to write model checkpoints to")
    args = parser.parse_args()
    train_models(models, args.data_path, args.train_path)
