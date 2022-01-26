# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:52:53 2022

@author: esm
"""
# from fastapi import FastAPI

# # Instantiate the app.
# app = FastAPI()

# # Define a GET on the specified endpoint.
# @app.get("/")
# async def say_hello():
    # return {"greeting": "Hello World!!!"}
    
# # To run our app, we will use uvicorn in our shell: uvicorn main:app --reload (uvicorn test_fastapi:app --reload). 
# # By default, our app will be available locally at http://127.0.0.1:8000. 
# # The --reload allows you to make changes to your code and have them
# # instantly deployed without restarting uvicorn.


#%%

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list] 
    item_id: int

app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/items/")
async def create_item(item: TaggedItem):
    return item