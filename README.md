# dudoo_api
A simple ai chat bot api build in django 

This chatbot trained on a covid-19 question and answers dataset

Before running the server make sure you have
  - pytorch
  - numpy
  - nltk
  - django
  - djangorestframework
 
 Dataset and model are in "./chat/ai/"
 
 For training dataset
  Run
    - python train.py (make sure your in "./chat/ai/" directory)
    
 For start server
  Run
    - python manage.py runserver <your_ip_address>:8000
    
  Test:
    - http://<your_ip_address>:8000/ai/?q=hai
    
    - http://<your_ip_address>:8000/ai/?q=what+is+covid+19
