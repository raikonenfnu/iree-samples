
import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite4/uint8/2?lite-format=tflite"

class EfficientNetQuantTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(EfficientNetQuantTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(EfficientNetQuantTest, self).compare_results(iree_results, tflite_results, details)

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

