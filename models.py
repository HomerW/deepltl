import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping1D, Input, Reshape
from LTLOperator import LTLOperator

trace_length = 15
num_variables = 3
batch_size = 100

def get_model_zero():
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, trace_length, num_variables)))
    model.add(LTLOperator(num_variables, 1, trace_length, metric=False))
    model.add(Cropping1D((0, trace_length-1)))
    model.add(Reshape((-1,1)))
    return model

def get_model_one():
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, trace_length, num_variables)))
    model.add(LTLOperator(num_variables, 3, trace_length, metric=False))
    model.add(LTLOperator(3, 1, trace_length, metric=False))
    model.add(Cropping1D((0, trace_length-1)))
    model.add(Reshape((-1,1)))
    return model

def get_model_two():
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, trace_length, num_variables)))
    model.add(LTLOperator(num_variables, 5, trace_length, metric=False))
    model.add(LTLOperator(5, 5, trace_length, metric=False))
    model.add(LTLOperator(5, 1, trace_length, metric=False))
    model.add(Cropping1D((0, trace_length-1)))
    model.add(Reshape((-1,1)))
    return model
