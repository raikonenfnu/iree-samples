
import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3?lite-format=tflite"

# Note this one takes forever right now. Great for performance work!
class BirdClassifierQuantTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(BirdClassifierQuantTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(BirdClassifierQuantTest, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=5e-3).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

