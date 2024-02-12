import pandas as pd
from math import ceil
from transformers import pipeline
import os
import sys

def postprocess(dataset_name, output_file):
  res= []
  NUM_SPLITS = 3
  NUM_SPLITS-=1
  generator = pipeline("text-generation", model="gpt2")

  for i,filename in enumerate(os.listdir(dataset_name)):
    print(filename)
    with open(os.path.join(dataset_name, filename), "r", encoding="latin1") as file:
      string = file.read()

      if string=="":
        continue

      conversation = (string.split("###\nConversation")[-1].strip())
      split_conversation = (conversation.replace("Student: ","##SEP##").replace("Teacher: ","##SEP##").split("##SEP##")[1:])
      sep_conversation = ["Teacher: "+i if not i.startswith("Student") else i for i in conversation.split("Teacher: ")][:-1]

      for i in range(NUM_SPLITS+1):
        index = (ceil(((len(sep_conversation)-1)/NUM_SPLITS)*i))
        prompt = ("".join(sep_conversation[0:index+1]))
        answer = ("Teacher: "+split_conversation[min((index*2)+1,len(split_conversation)-1)])
        res.append({"Prompt":prompt, "Bad answer":"", "Good answer":answer})

    if(i>10):
      break

  pd.DataFrame(res).to_csv('dataset.csv', index=False)


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python script.py <dataset_name> <output_file>")
    sys.exit(1)

  dataset_name = sys.argv[1]
  output_file = sys.argv[2]

  postprocess(dataset_name, output_file)