# -*- coding: utf-8 -*-
# @Time : 2020/2/6 5:42 PM 
# @Author : qiujiafa
# @File : app.py

from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_url_path='/static')


@app.route('/message', methods=['POST'])
def reply():
    request_msg = request.form['msg']
    print(request_msg)
    answer = 'hello'

    return jsonify({'text': answer})


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9996)
