import tensorflow as tf
import time
from transformers import BertModel, BertTokenizer, TFBertModel

tf_model = TFBertModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased", from_pt=True)
tokenizer = BertTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=512)
for key in encoded_input:
    encoded_input[key] = tf.expand_dims(tf.convert_to_tensor(encoded_input[key]),0)
warmup = 1
total_iter = 10
num_iter = total_iter - warmup
for i in range(total_iter):
    output = tf_model(encoded_input)
    if(i == warmup-1):
        start = time.time()

end = time.time()
total_time = end - start
print(output)
print("time: "+str(total_time))
print("time/iter: "+str(total_time/num_iter))
