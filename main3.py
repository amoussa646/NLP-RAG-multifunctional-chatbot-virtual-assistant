import os
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import faiss
import numpy as np
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration,AutoTokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow

# Initialize retriever and model


tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
retriever = AutoTokenizer.from_pretrained("facebook/rag-token-base")
model = AutoTokenizer.from_pretrained("facebook/rag-token-base")

#Load the tokenizers separately
generator_tokenizer = tokenizer
# Initialize RAG tokenizer
# Initialize tokenizers
question_encoder_tokenizer=tokenizer
import torch

input_text = "what did Abdallah study before computer engineering?"
input_ids = question_encoder_tokenizer(input_text, return_tensors="pt").input_ids

# Use the model to generate an answer
with torch.no_grad():
    generated = model(input_id)


# Decode the generated answer
generated_text = question_encoder_tokenizer.batch_decode(generated, skip_special_tokens=True)
print(generated_text)


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Documents to embed and index

documents = [ '''Resume

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

Skills:
- Programming Languages:Python, Java, JavaScript, Dart
- Frameworks: Django, FastAPI, Flask, Flutter, MERN stack
- PostgreSQL, MongoDB
-RESTful API
- IoT: Tasmota ﬁrmware, ESP boards, InﬂuxDB, Grafana, Telegraf
- Unity
-Docker, Github
- Linux
AI/ML: CV, NLP, PyTorch

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
''', '''Abdallah Moussa
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
''']

# Compute embeddings
document_embeddings = embed_model.encode(documents)

# Create FAISS index
index = faiss.IndexFlatL2(384)
index.add(np.array(document_embeddings))

# Query the FAISS index
query = "What did Abdallah study before Computer Engineering?"
query_embedding = embed_model.encode([query])
distances, indices = index.search(np.array(query_embedding), k=1)

# Retrieve the top result
print("Top result:", documents[indices[0][0]])

# Retrieve relevant documents
query = "What did Abdallah study before Computer Engineering?"
query_embedding = embed_model.encode([query])
distances, indices = index.search(np.array(query_embedding), k=1)
retrieved_document = documents[indices[0][0]]

# Generate answer using RAG
input_text = f"{query} {retrieved_document}"
input_ids = question_encoder_tokenizer(input_text, return_tensors="pt").input_ids

with torch.no_grad():
    generated = model.generate(input_ids, num_beams=5, num_return_sequences=1)

generated_text = question_encoder_tokenizer.batch_decode(generated, skip_special_tokens=True)
print(generated_text)
