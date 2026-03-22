import os, sys, requests, json

password = "admin123"
API_KEY = "sk-hardcoded-secret-key"

def get_user(id):
    query = "SELECT * FROM users WHERE id = " + id
    r = requests.get("http://api.example.com/users/" + id, verify=False)
    data = json.loads(r.text)
    return data

def process(items):
    result = []
    for i in range(len(items)):
        result = result + [items[i] * 2]
    return result

def divide(a, b):
    return a / b
