import model_helper
import data_helper
import argparse

parser = argparse.ArgumentParser(description = 'Predict imag')
parser.add_argument('input', nargs = '?', action = "store", default = "flowers/test/1/image_06752.jpg")
parser.add_argument('checkpoint', nargs = '?', action = "store", default = "./checkpoint.pth")
parser.add_argument('--top_k', dest = 'top_k', nargs = '?', action = "store", type = int, default = 3)
parser.add_argument('--category_names', dest = 'cat', nargs = '?', action = "store", default = 'cat_to_name.json')
parser.add_argument('--gpu', dest = 'gpu', nargs='?', action="store", default='GPU')

parsed_args = parser.parse_args()
input_path = parsed_args.input
checkpoint = parsed_args.checkpoint
topk = parsed_args.top_k
cat = parsed_args.cat
gpu = parsed_args.gpu

model =  model_helper.load_model(path = checkpoint)
flower, probability = model_helper.predict(input_path, model, topk, power = gpu, category_names = cat)