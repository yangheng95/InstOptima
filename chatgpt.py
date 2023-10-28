# -*- coding: utf-8 -*-
# file: chatgpt.py
# time: 13:09 16/05/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.

import json
import logging
import os
import time
from hashlib import sha256

import openai


class Chatbot:
    def __init__(self, system_prompt=None, model='gpt-3.5-turbo', log_dir='logs', api_key=None, **kwargs):
        print(os.environ)
        try:
            assert os.environ.get("OPENAI_API_KEY"), "Please set OPENAI_API_KEY environment variable."
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        except Exception as e:
            print("Please set OPENAI_API_KEY environment variable.")
            openai.api_key = api_key
        self._chatbot = None
        self._model = model
        if system_prompt:
            self.system_prompts = [
                {'role': 'system', 'content': system_prompt},
            ]
        else:
            self.system_prompts = [
                {'role': 'system', 'content': 'You are ChatGPT, a very helpful assistant.'},
            ]
        self.messages = []

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(
            logging.FileHandler('{}/chatbot.log'.format(
                log_dir)
            )
        )
        self._logger.setLevel(logging.INFO)

    def chat(self,
             prompt,
             max_tokens=500,
             temperature=1,
             top_p=1,
             frequency_penalty=0,
             presence_penalty=0.6,
             stop=['<EndofConversation>'],
             **kwargs
             ):
        if kwargs.get('system_prompt'):
            self.system_prompts = [
                {'role': 'system', 'content': kwargs.get('system_prompt')},
            ]
        _messages = [
            {"role": "user", "content": prompt
        }]
        _response = openai.ChatCompletion.create(
            model=self._model,
            messages=self.system_prompts + _messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        info = json.dumps(
            {
                'tag': kwargs.get('tag', 'Debug'),
                'prompt': prompt,
                'response': _response.choices[0].message.content,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
        )
        # print(info)
        self._logger.info(info)
        return _response.choices[0].message.content

    def chat_with_history(self,
                          prompt,
                          max_tokens=500,
                          temperature=1,
                          top_p=1,
                          frequency_penalty=0,
                          presence_penalty=0.6,
                          stop=['<EndofConversation>'],
                          **kwargs
                          ):
        if kwargs.get('system_prompt'):
            self.system_prompts = [
                {'role': 'system', 'content': kwargs.get('system_prompt')},
            ]
        self.messages.append({"role": "user", "content": prompt})

        _response = openai.ChatCompletion.create(
            model=self._model,
            messages=self.system_prompts + self.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        self.messages.append(dict(_response.choices[0].message))
        info = json.dumps(
            {
                'tag': kwargs.get('tag', 'Debug'),
                'prompt': prompt,
                'response': _response.choices[0].message.content,
                'messages': self.messages,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
        )
        # print(info)
        self._logger.info(info)
        return _response.choices[0].message.content


if __name__ == '__main__':
    import flask
    from flask import request, jsonify

    app = flask.Flask(__name__)

    chatbot = Chatbot(system_prompt=None, model='gpt-3.5-turbo-0613')


    @app.route('/chat', methods=['POST'])
    def chat():

        # try:
            if not request.args.get('password'):
                return jsonify({"error": "No password"})

            if sha256(request.args.get(
                    'password').encode()).hexdigest() != "01cb92dfff4091c2bee0f343b2af049fb39b45c08a1e5132b834e12e037d919d":
                return jsonify({"error": "Wrong password"})
            prompt = request.args.get('prompt')

            print('message_history', request.args.getlist('message_history'))
            if request.args.getlist('message_history'):
                chatbot.messages = eval(request.args.get('message_history'))
                print(chatbot.messages)
            else:
                chatbot.messages = []

            response = chatbot.chat_with_history(prompt, tag=request.args.get('tag'), max_tokens=2000)
            return jsonify({"response": response, "message_history": json.dumps(chatbot.messages)})
        # except Exception as e:
        #     return jsonify({"error": str(e)})


    app.run(port=6789, debug=False)

    # TEST FLASK API
    # curl -i -H "Content-Type: application/json" -X POST -d '{"prompt":"Hello, my name is"}' http://localhost:6789/chat
