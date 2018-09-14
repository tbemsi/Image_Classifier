import argparse
import json
import torch
from PIL import Image
from utils import process_image, load_checkpoint
from collections import OrderedDict
import numpy as np

parser = argparse.ArgumentParser(description = 'Predict the type of a flower')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint', default = '/home/workspace/aipnd-project/classifier.pth')
parser.add_argument('--image_path', type=str, help='Path to file', default = 'flowers/test/28/image_05320.jpg')
parser.add_argument('gpu', type=bool, default=True, help='Whether to use GPU during inference or not')
parser.add_argument('--topk', type=int, help = 'Number of k to predict', default=5)
parser.add_argument('--cat_to_name_json', type=str, help='Json file to load for class values to name conversion', default='cat_to_name.json')
args = parser.parse_args()

image_path = args.image_path
with open(args.cat_to_name_json, 'r') as f:
    cat_to_name = json.load(f)
   
#Loading checkpoint
model, checkpoint = load_checkpoint(args.checkpoint)

#Process PIL image
img = process_image(image_path)

def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

if args.topk:
    probs, top_labels, top_flowers = predict(image_path, model, args.topk, 'cuda' if args.gpu else 'cpu')
    print('Probabilities of top {} flowers:'.format(args.topk))
    for i in range(args.topk):
        print('{} : {:.2f}'.format(top_flowers[i], probs[i]))
    else:
        probs, flower_names = predict(image_path, model)
        print('Prediction is {} with {:.2f} probability'.format(flower_names[0], probs[0]))



