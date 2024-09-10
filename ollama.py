import ollama

message = """

実行したいプロンプトを記載する

"""

response = ollama.chat(model='<model名>', messages=[
  {
    'role': 'user',
    'content': message,
  },
])
print(response['message']['content'])
