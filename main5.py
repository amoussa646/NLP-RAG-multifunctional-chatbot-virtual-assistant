import nltk
nltk.download('punkt')

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key='')

from openai import OpenAI
openai_api = OpenAI(api_key="")

# STORE DATA

## CREATE INDEX

INDEX_NAME = "moses-info"
NAMESPACE = "DOCUMENTATION_OPENAI"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MODEL_SIZE = 3072
# run once
pc.create_index(
    name=INDEX_NAME,
    dimension=EMBEDDING_MODEL_SIZE,
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)


# GET DATA

#Produce single corpus text

### FUNCTIONS
#https://en.wikipedia.org/wiki/Special:Random/*math*

import requests
from bs4 import BeautifulSoup

http_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def get_text_from_website(url):
    response = requests.get(url, headers=http_headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    text_content = soup.get_text()

    return text_content

def get_text_from_elements(url):
    response = requests.get(url, headers=http_headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    text_content = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text_content

def get_text_from_tables(url):
    response = requests.get(url, headers=http_headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table, you might need to add specific attributes to target the right table
    table = soup.find('table')  # Add attributes if necessary, like {'class': 'my-table-class'}

    # Initialize a list to store all rows of data
    table_data = []

    # Extract headings (optional, if the table includes header rows)
    headers = []
    header_tags = table.find_all('th')
    for th in header_tags:
        headers.append(th.get_text(strip=True))

    # Extract rows
    for row in table.find_all('tr'):
        # Extract columns
        cols = row.find_all('td')
        row_data = [ele.text.strip() for ele in cols]
        if row_data:  # Ensure the row has data
            table_data.append(row_data)

    text_content = ' '.join([' & '.join(sublist) for sublist in table_data])

    return f""". {text_content}."""


def collapse_list_to_string(array_strings):
    corpus = ""
    for article in array_strings:
        corpus += article
    return corpus


# website_url = input("Enter the web URL: ")
# website_text = get_text_from_website(website_url)
# print(website_text)

from newspaper import Article

def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

import re

def count_tokens(text):
  return len(text.split())

def count_words(text):
  words = re.findall(r'\w+', text)
  return len(words)

def print_stats(text):
    token_count = count_tokens(text)
    word_count = count_words(text)

    print("Number of tokens:", token_count)
    print("Number of words:", word_count)

### MAIN

import pprint

links = []
links.append('https://openai.com/blog/new-embedding-models-and-api-updates')
# links.append('https://react.dev/learn')

articles = []
for url in links:
    #all_tables_as_sentence = get_text_from_tables(url)
    all_sentences = get_text_from_elements(url)
    articles.append(all_sentences)
    #articles.append(all_tables_as_sentence)




corpus = collapse_list_to_string(articles)
# new corpus my info

corpus =  '''Resume

Abdallah Moussa

A dedicated tech explorer on a perpetual journey of innovation,embracing every project as a chance to make a difference.

Experience:

Neuss, Irbid, Jordan — Remote Software Engineer  
October 2022 - November 2023
- Developed a comprehensive time tracking app, incorporatingUser Authentication and Account Management for employees torecord their activity and achievements using Flutter.
- Implemented real-time chatting functionality to facilitate seamless communication among users using Flutter.
- Created backend logic for User Authentication and Account Management using FastAPI.
- Optimized projects through dockerization, enabling remote deployment of machine learning models for enhanced efficiency and accessibility.
- Adapted Tasmota ﬁrmware for easy ESP board connections, conﬁguration, data collection from sensors, and efficient data transmission to InﬂuxDB and Grafana via Telegraf.
- Applied machine learning techniques, primarily with PyTorch, for IoT data analysis, expanding the company's AI-IoT capabilities.
- Fine-tuned the Whisper model using deep learning techniques, resulting in the development of sophisticated dialogue systems, catering to various Arabic dialects.



EDUCATION: 
German University in Cairo  in B
.Sc. Computer Engineering GPA: 2.49 
October 2014 - February 2024 (2016 gap year - personal projects)
Studied Mechatronics initially, then switched to Computer Engineering.

Social Companion Robot - Bachelor Thesis
- Computer Vision: Implemented face, emotion, object and colorrecognition, image captioning.
- NLP: Developed a conversational chatbot using PyTorch, integrated with a customized chatbot for versatile user interactions.- Speech-to-text and Text-to-speech engines.



Relative Coursework
Artiﬁcial Intelligence
Machine learning
Human Computer Interaction
Computer Vision
Deep learning 
Human Computer Interaction
Business of Software
Computer & Network Security

Languages:
Arabic: Native
English: Proﬁcient
German: Beginner

A computer engineer with a background in Mechatronics, and a primary passion for AI & Robotics. On a perpetual journey of innovation, embracing each project as an opportunity to make a difference, leveraging expertise from diverse domains.

Who I am
I began studying Mechatronics engineering to build robots, quickly excelling in mechanics and electronics. Realizing the need for advanced programming to make my creations smart and useful, I switched to Computer Engineering.

My primary focus is machine learning due to its pivotal role in robotics. Along the way, I became proficient in Flutter for app development and FastAPI for backend development due to always depending on them in my personal projects and my last work experience.

Skills
HTML
CSS
JS
Python
Java
Dart
C#
Flutter
React
FastApi
Flask
MERN stack
Selenium
Matlab
Embedded Systems
Unity
WolframAlpha
Numpy
scikit-learn
OpenCV
Pytorch
Tensorflow
MongoDB
InfluxDb
MySQL
PostgreSQL
Firebase
Git
Docker
Ubuntu
HTML
CSS
JS
Python
Java
Dart
C#
Flutter
React
FastApi
Flask
MERN stack
Selenium
Matlab
Embedded Systems
Unity
WolframAlpha
Numpy
scikit-learn
OpenCV
Pytorch
Tensorflow
MongoDB
InfluxDb
MySQL
PostgreSQL
Firebase
Git
Docker
Ubuntu
Education
Secondary Education - Allthanaweya Al-Amma
Educational Home School - Dar Al-Tarbiah
96.7%
Switched to Computer Engineering in Semester 7
Mechatronics Engineering
The German University in Cairo
2018 - 2024
Computer Engineering
The German University in Cairo
GPA (German): 2.49
Experience
October 2022-October 2023
Remote Software Engineer
Neuss for App development
Flutter - FastApi - PyTorch - PostgreSQL - InfluxDB - Telegraf - Grafana
Testimonials


Services
These are some of the services I offer. Reach out to me if I can help you with any!

Machine Learning
Computer Vision
Flutter Multi-Platform App Development
Backend Development
Unity Game Development & AR/VR
IoT & AIoT Solutions
Embedded Systems
Projects
Device Price Classification and Management System (Machine Learning + Web Application)
Devices Price Classification System with functionality for adding, deleting, viewing, and editing devices, using Python, Flask and Spring Boot.
Machine Learning
Flask
Spring Boot
Python
Java
Full-Stack
Computer Vision tasks for robotics
OpenCV - PyTorch - Computer Vision - FastApi -Python - CNN - RNN - YoloV5 - Diffrential Robot - Robot Arm - ResNet50
Image Captioning
Person/Face/Eyes/Smile/Object/Color Detection
Human Follower
Rock Paper Scissors
Object Tracker
sessions tracker + Chat platform for company Employees (Flutter/FastApi)
A Flutter app with a FastApi backend that allows employees of a company to record the start and end time of their working sessions along with the tasks completed with option for breaks and an option to visit old sessions. Additiong there is a chatting platform that includes all the employees of the company using MQTT
Flutter
FastApi
PostgreSQL
RESTful API
MQTT
Real-Time texting
.
Unity Endless Energy Orbs runner
An endless runner game using Unity where the runner is an a white Orb that have to avoid obstacles and can collect other energy orbs to use their powers , Red , Green and Blue.
Unity
C#
Game Development
Airlines Company tickets booking system (MERN stack)
An Airlines website,where users can search for flights, make reservations, complete payments, and admins can manage flights info
MongoDB
Express
React
Node
Full-Stack
''' 

print_stats(corpus)
print(corpus)


# EMBEDDING

#Make dataframe, get embeddings, store dataframe, chunk, get objects for upserting

#Upsert

### FUNCTIONS

import pandas as pd
from tabulate import tabulate
import json
import nltk
from nltk.tokenize import sent_tokenize

# COLUMNS: IDs, Paragraphs, Embeddings

PARAGRAPH_COLUMN = "Paragraph"
ID_COLUMN = "ID"
EMBEDDING_COLUMN = "Embedding"
CHUNK_SIZE = 5
ADJACENT_SIZE = 1


def remove_empty_strings(strings):
  non_empty_strings = []
  for string in strings:
    if string:
      non_empty_strings.append(string)
  return non_empty_strings


def partition_list(array_strings, n):
  return [array_strings[i:i + n] for i in range(0, len(array_strings), n)]


def corpus_to_dataframe(text):
    sentences = sent_tokenize(text)

    partitioned_list = partition_list(sentences, CHUNK_SIZE)

    paragraphs = []

    for partition in partitioned_list:
        corpus = collapse_list_to_string(partition)
        paragraphs.append(corpus)

    df = pd.DataFrame(paragraphs, columns=[PARAGRAPH_COLUMN])
    return df


def add_ids_column(df):
    df = df.reset_index()
    df.rename(columns={"index": ID_COLUMN}, inplace=True)


def print_dataframe(df):
    print(tabulate(df, headers="keys", tablefmt="grid"))


def get_embedding(text):
    response = openai_api.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def add_embeddings_column(df):
    embeddings = []

    for sentence in df[PARAGRAPH_COLUMN]:
        embedding_object = get_embedding(sentence)
        embeddings.append(embedding_object)

    df[EMBEDDING_COLUMN] = embeddings


def get_objects(df):
    objects = df.to_dict(orient='records')

    pinecone_objects = []
    for index, raw_object in enumerate(objects):
        previous = get_previous_items(objects, index)
        next = get_next_items(objects, index)

        pinecone_object = {
            "id": f"{index}",
            "values": raw_object[EMBEDDING_COLUMN],
            "metadata": {
                "text": raw_object[PARAGRAPH_COLUMN],
                "text_source": links[0]
                }
        }
        pinecone_objects.append(pinecone_object)

    return pinecone_objects


def default(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


def clean_text(text):
  text = text.replace('\n', ' ')
  text = ' '.join(text.split())
  return text


def save_dataframe(df, filename='dataframe_export', file_format='csv'):
  if file_format == 'csv':
    df.to_csv(filename + '.csv', index=False)
  elif file_format == 'json':
    df.to_json(filename + '.json', orient='records')
  else:
    raise ValueError(f"Unsupported file format: {file_format}")

  print(f"Dataframe saved to '{filename}.{file_format}'")

def get_string_from_array(array_strings):
    sum = ""
    for item in array_strings:
        sum += f"""{item}, """
    return sum

def get_previous_items(array_strings, current_index):
    start_prev = max(0, current_index - ADJACENT_SIZE)
    previous_items = array_strings[start_prev:current_index]

    return get_string_from_array(previous_items)

def get_next_items(array_strings, current_index):
    end_next = min(len(array_strings), current_index + (ADJACENT_SIZE + 1))
    next_items = array_strings[current_index + 1:end_next]

    return get_string_from_array(next_items)

### MAIN

df = corpus_to_dataframe(corpus)
add_ids_column(df)
add_embeddings_column(df)
save_dataframe(df)
print_dataframe(df)

objects = get_objects(df)

## UPSERT VECTORS

index = pc.Index(INDEX_NAME)

index.upsert(
    namespace=NAMESPACE,
    vectors=objects
)

## CHECK UPDATE

index.describe_index_stats()


# SEARCH

### FUNCTIONS

from openai import OpenAI
client_openai = OpenAI(api_key="")

llm_setting = "openai"
# llm_setting = "anthropic"

import anthropic
client_anthropic = anthropic.Anthropic(api_key="")

def call_llm(llm_setting, systemPrompt, text):
    if llm_setting == "openai":
        completion = client_openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": text}
            ]
        )
        return completion.choices[0].message.content
    elif llm_setting == "anthropic":
        message = client_anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.5,
            system=systemPrompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        )
        return message.content
    else:
        return "error"

def call_weak_llm(llm_setting, systemPrompt, text):
    if llm_setting == "openai":
        completion = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": text}
            ]
        )
        return completion.choices[0].message.content
    elif llm_setting == "anthropic":
        message = client_anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.5,
            system=systemPrompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        )

        return message.content
    else:
        return "error"
    

# USER_QUESTION = "What's this document about?"
# # new user question
# USER_QUESTION = "What did Abdallah study before Computer Engineering?"

def search(user_question):

    pre_query_prompt = f"""HERE IS A USER PROMPT:
    {user_question}
    --
    GIVEN THE USER QUESTION, WHATS A SINGLE WORD OR SINGLE PHRASE THAT ENCAPSULATES THE USER INTENT
    """
    print(pre_query_prompt)

    search_query = call_llm(llm_setting, "You are doing finding a search query", pre_query_prompt)
    print(search_query)

    vector_query = get_embedding(search_query)

    result = index.query(
    namespace=NAMESPACE,
    vector=vector_query,
    top_k=3,
    include_values=False,
    include_metadata=True
    )
    print(result['matches'][0])
    return result['matches'][0]



# RAG

### MAIN
def RAG_Personal(user_question):
    rag_prompt = f"""
    HERE IS THE CONTEXT:
    {search(user_question=user_question)}
    --
    GIVEN THE CONTEXT, ANSWER THE FOLLOWING QUESTION:
    {user_question}
    """
    print(rag_prompt)

    answer = call_llm(llm_setting, "You are doing Retrieval Augmented Generation. FYI.", rag_prompt)
    print(answer)
    return answer



pc.delete_index(INDEX_NAME)




