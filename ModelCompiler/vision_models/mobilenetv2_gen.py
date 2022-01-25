from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import os

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

if __name__ == "__main__":
    HUB_PATH = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    model_path = hub.resolve(HUB_PATH)
    loaded_model = tf.saved_model.load(model_path)

    call = loaded_model.__call__.get_concrete_function(tf.TensorSpec([1, 224, 224, 3], tf.float32))
    signatures = {'predict': call}
    resaved_model_path = '/tmp/resaved_model'
    tf.saved_model.save(loaded_model, resaved_model_path, signatures=signatures)

    # Compile the model using IREE
    compiler_module = tfc.compile_saved_model(
                        resaved_model_path,
                        import_type="SIGNATURE_DEF",
                        exported_names = ["predict"],
                        saved_model_tags=set(["serve"]),
                        import_only=True)

    # Save module as MLIR file in a directory
    ARITFACTS_DIR = os.getcwd()
    mlir_path = os.path.join(ARITFACTS_DIR, "mobilenetv2.mlir")
    with open(mlir_path, "wt") as output_file:
        output_file.write(compiler_module.decode('utf-8'))
    print(f"Wrote MLIR to path '{mlir_path}'")