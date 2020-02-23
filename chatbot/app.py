# -*- coding: utf-8 -*-
# @Time : 2020/2/6 5:42 PM 
# @Author : qiujiafa
# @File : app.py

from flask import Flask, render_template, request, jsonify

from chatbot.chatbot_main import Chatbot

app = Flask(__name__, static_url_path='/static')
chatbot = Chatbot()


@app.route('/message', methods=['POST'])
def reply():
    question = request.form['msg']
    print(question)
    answer = chatbot.answer(question)

    return jsonify({'text': answer})


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9996)
