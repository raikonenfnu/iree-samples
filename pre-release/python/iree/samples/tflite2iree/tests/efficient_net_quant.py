
import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite4/uint8/2?lite-format=tflite"

# Note this one takes forever right now. Great for performance work!
class EfficientNetQuantTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(EfficientNetQuantTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(EfficientNetQuantTest, self).compare_results(iree_results, tflite_results, details)
    a = tflite_results[0].astype(float)
    b = iree_results[0].astype(float)
    self.assertTrue((numpy.abs(a-b) <= 1).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

