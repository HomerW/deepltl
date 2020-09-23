import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping1D, Input, Reshape
import numpy as np
from LTLOperator import LTLOperator
from cs import generate_cs
from translation import translate
from train_utils import Anneal, DiscreteAcc
import spot

def get_model():
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, trace_length, num_variables)))
    model.add(LTLOperator(num_variables, 1, trace_length, metric=False))
    model.add(Cropping1D((0, trace_length-1)))
    model.add(Reshape((-1,1)))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    formula = "a U b"
    lits = ['a', 'b', 'c']
    num_train = 500
    trace_length = 15
    num_variables = len(lits)
    batch_size = 100
    model = get_model()

    checkpoint_path = "training/cp.ckpt"

    pos_train, neg_train = generate_cs(formula, lits, trace_length, num_train, num_train)
    train_labels = tf.concat([tf.ones((num_train,)), tf.zeros((num_train,))], 0)
    train_traces = tf.convert_to_tensor(pos_train + neg_train, dtype=tf.float32)

    history = model.fit(train_traces, train_labels,
                        epochs=3000, batch_size=batch_size,
                        verbose=1,
                        shuffle=True,
                        callbacks=[Anneal(), DiscreteAcc(train_traces, train_labels)])

    model.save_weights(checkpoint_path)

    layer_weights = [l.get_weights() for l in model.layers[:-2]]

    spot_lits = [spot.formula("a"), spot.formula("b"), spot.formula("c")]
    learned_formula = translate(layer_weights, spot_lits, metric=False)
    print(f"Learned formula: {learned_formula}")
