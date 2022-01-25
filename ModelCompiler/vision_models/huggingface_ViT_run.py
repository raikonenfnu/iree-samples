from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import sys
from absl import app
import os

import tensorflow as tf
from transformers import ViTFeatureExtractor, TFViTForImageClassification
from PIL import Image
import requests


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
    # Prepping Data
    dummy_image = tf.random.uniform(shape=[1, 3, 224, 224], minval = -1.0, maxval =1.0, dtype=tf.float32)

    # If want to test real data
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    inputs = feature_extractor(images=image, return_tensors="tf")

    # Compile the model using IREE
    compiler_module = tfc.compile_module(ViTModule(), exported_names = ["predict"], import_only=True)
    flatbuffer_blob = compile_str(compiler_module, input_type="mhlo", target_backends=["dylib-llvm-aot"])

    # Save module as MLIR file in a directory
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    tracer = ireert.Tracer(os.getcwd())
    config = ireert.Config("dylib",tracer)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ViTCompiled = ctx.modules.module
    result = ViTCompiled.predict(dummy_image)
    print(result)
    predicted_class_idx = tf.math.argmax(result, axis=-1)[0]
    print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
