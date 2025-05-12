import textwrap
import json
from transformers import pipeline
import sentencepiece
import torch

def load_data(data):
  with open(data) as data_f:
    unsplit_data = json.load(data_f)
  data_array = dict([])

  for elt in unsplit_data:
    mTitle = (elt["title"])
    mReview = ""
    for review in elt["reviews"]:
      mReview = mReview + " " + ((review["text"]))
    data_array[mTitle] = mReview
  return data_array

def print_summary(summaries, movie):
  wrapper = textwrap.TextWrapper(width=80, break_long_words=False, break_on_hyphens=False)
  print(movie, "generated summary:\n", wrapper.fill(summaries[movie]), "\n")

if __name__ == '__main__':
  data = load_data("HLT_data.json")

  summarization_pipeline = pipeline("summarization", model="abhiramd22/t5-base-finetuned-to-summarize-movie-reviews")
  summary = dict([])
  i=0

  for movie in data:
    output = summarization_pipeline(data[movie], clean_up_tokenization_spaces=True, max_length=256)
    summary[movie] = output[0]['summary_text']
    print("movie", movie, "complete")
    i+=1
    if(i==5):
      break

  i=0
  for movie in data:
    print_summary(summary, movie)
    i+=1
    if(i==5):
      break



