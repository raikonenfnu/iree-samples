from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import os

import tensorflow as tf
from transformers import ViTFeatureExtractor, TFViTForImageClassification


# Create inputs shape [BxCxWxH].
image_input = [tf.TensorSpec(shape=[1, 3, 224, 224], dtype=tf.float32)]

class ViTModule(tf.Module):
    def __init__(self):
        super(ViTModule, self).__init__()
        # Create a ViT network.
        self.m = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m.predict = lambda x: self.m.call(x, training=False)

    @tf.function(input_signature=image_input)
    def predict(self, input_image):
        return self.m.predict(input_image)

if __name__ == "__main__":
    # Compile the model using IREE
    compiler_module = tfc.compile_module(ViTModule(), exported_names = ["predict"], import_only=True)
    # Save module as MLIR file in a directory
    ARITFACTS_DIR = os.getcwd()
    mlir_path = os.path.join(ARITFACTS_DIR, "vit.mlir")
    with open(mlir_path, "wt") as output_file:
        output_file.write(compiler_module.decode('utf-8'))
    print(f"Wrote MLIR to path '{mlir_path}'")