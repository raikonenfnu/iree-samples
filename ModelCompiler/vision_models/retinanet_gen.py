from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import os

import tensorflow as tf
from official.vision.beta.modeling import retinanet_model

# Create inputs shape [BxCxWxH].
image_input = [tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)]

def CreateRetinaNet():
    num_classes = 3
    min_level = 3
    max_level = 7
    num_scales = 3
    aspect_ratios = [1.0]
    num_anchors_per_location = num_scales * len(aspect_ratios)

    backbone = resnet.ResNet(model_id=50)
    decoder = fpn.FPN(
        input_specs=backbone.output_specs,
        min_level=min_level,
        max_level=max_level)
    head = dense_prediction_heads.RetinaNetHead(
        min_level=min_level,
        max_level=max_level,
        num_classes=num_classes,
        num_anchors_per_location=num_anchors_per_location)
    generator = detection_generator.MultilevelDetectionGenerator(
        max_num_detections=10)
    model = retinanet_model.RetinaNetModel(
        backbone=backbone,
        decoder=decoder,
        head=head,
        detection_generator=generator,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=3)

class SSDMobileNetV2Compiled(tf.Module):
    def __init__(self):
        super(SSDMobileNetV2Compiled, self).__init__()
        # Create a ViT network.
        self.m = RetinaNetModel(module_handle).signatures['default']

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m_predict = lambda x: self.m(x, training=False)

    @tf.function(input_signature=image_input)
    def predict(self, input_image):
        return self.m_predict(input_image)

if __name__ == "__main__":
    # Prepping Data
    dummy_image = tf.random.uniform(shape=[1, 224, 224, 3], minval = 0.0, maxval =1.0, dtype=tf.float32)

    # If want to test real data
    # Compile the model using IREE
    compiler_module = tfc.compile_module(SSDMobileNetV2Compiled(), exported_names = ["predict"], import_only=True)
    flatbuffer_blob = compile_str(compiler_module, input_type="mhlo", target_backends=["dylib-llvm-aot"])

    # Save module as MLIR file in a directory
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    tracer = ireert.Tracer(os.getcwd())
    config = ireert.Config("dylib",tracer)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    SSDMobileNetV2Compiled = ctx.modules.module
    result = SSDMobileNetV2Compiled.predict(dummy_image)
    print(result)
