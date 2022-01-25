from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import os

import tensorflow as tf

# Create inputs shape [BxCxWxH].
INPUT_SHAPE = [1, 224, 224, 3]
image_input = [tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)]
# Initialize model
# Need TF-Nightly
tf_model = tf.keras.applications.regnet.RegNetX016(
    model_name='regnetx016', include_top=True, include_preprocessing=True,
    weights='imagenet', input_tensor=None, input_shape=tuple(INPUT_SHAPE[1:]), pooling=None,
    classes=1000, classifier_activation='softmax')

class ResNetModule(tf.Module):

  def __init__(self):
    super(ResNetModule, self).__init__()
    self.m = tf_model
    self.m.predict = lambda x: self.m.call(x, training=False)
    self.predict = tf.function(
            input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)])(tf_model.predict)

if __name__ == "__main__":
    # Compile the model using IREE
    compiler_module = tfc.compile_module(ResNetModule(), exported_names = ["predict"], import_only=True)
    # Save module as MLIR file in a directory
    ARITFACTS_DIR = os.getcwd()
    mlir_path = os.path.join(ARITFACTS_DIR, "resnet50.mlir")
    with open(mlir_path, "wt") as output_file:
        output_file.write(compiler_module.decode('utf-8'))
    print(f"Wrote MLIR to path '{mlir_path}'")