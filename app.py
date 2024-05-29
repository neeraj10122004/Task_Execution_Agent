from openai import OpenAI
from io import BytesIO
import IPython
import json
import os
from PIL import Image
import requests
import time
import streamlit as st
import google.generativeai as genai

st.title("LLM-Powered API Agent for Task Execution")
memory=[]

def Text_Generator(data):
    print("\n\ncalling text generator\n\n")
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = "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"
    )
    prompt=data["arguments"]["prompt"]
    print(prompt)
    completion = client.chat.completions.create(
    model="mistralai/mixtral-8x22b-instruct-v0.1",
    messages=[{"role":"user","content":prompt}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
    )

    ret=""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            ret+=chunk.choices[0].delta.content

    return ret
def Image_Generator(data):
    print("\n\ncalling image generator\n\n")
    STABILITY_KEY = "sk-2ySvipvf9ET7mqcgepz08kMobq4KLuFk2GlVGe4kLRQzHoOG"

    def send_generation_request(host,params,):
        headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
        }

        # Encode parameters
        files = {}
        image = params.pop("image", None)
        mask = params.pop("mask", None)
        if image is not None and image != '':
            files["image"] = open(image, 'rb')
        if mask is not None and mask != '':
            files["mask"] = open(mask, 'rb')
        if len(files)==0:
            files["none"] = ''

        # Send request
        print(f"Sending REST request to {host}...")
        response = requests.post(
            host,
            headers=headers,
            files=files,
            data=params
        )   
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        return response
            
    prompt = data["arguments"]["prompt"]
    negative_prompt = data["arguments"]["negative_prompt"]
    aspect_ratio = "1:1" #@param ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
    seed = 0 #@param {type:"integer"}
    output_format = "jpeg" #@param ["jpeg", "png"]

    host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"

    params = {
        "prompt" : prompt,
        "negative_prompt" : negative_prompt,
        "aspect_ratio" : aspect_ratio,
        "seed" : seed,
        "output_format" : output_format,
        "model" : "sd3",
        "mode" : "text-to-image"
    }

    response = send_generation_request(
        host,
        params
    )

    output_image = response.content
    finish_reason = response.headers.get("finish-reason")
    seed = response.headers.get("seed")

    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("Generation failed NSFW classifier")

    # Save and display result
    generated = f"generated_{seed}.{output_format}"
    with open(generated, "wb") as f:
        f.write(output_image)
    return generated

def Code_Generator(data):
    print("\n\ncalling code generator\n\n")
    prompt=data["arguments"]["prompt"]
    genai.configure(api_key = "AIzaSyDYfQmz7AHGbtvY4l5UJGVCa8JJgJrDjaQ")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(
        ["you are a coder write code in python based on the given prompt",prompt],
        generation_config=genai.types.GenerationConfig(
            temperature=0)
    )
    return response.text


def main(prompt):
    print(memory)
    data = """Your task is to generate a plan for the problem user gave using FUNCTIONS. Just generate the plan. Do not solve.
 
    <FUNCTIONS>[
        {
            "function": "ImageGenerator",
            "description": "Generates an Image based on a prompt description",
            "arguments": [
                {
                    "name": "prompt",
                    "type": "string",
                    "description": "Describe what is the key subject of the image, followed by the background."
                },
                {
                    "name": "negative_prompt",
                    "type": "string",
                    "description": "what shouldn't be in the image. Fill none if not specified."
                }
            ]
        },
        {
            "function": "CodeGenerator",
            "description": "Generates python code for a described problem",
            "arguments": [
                {
                    "name": "prompt",
                    "type": "string",
                    "description": "description of the problem for which the code needs to be generate"
                }
            ]
        },
        {
            "function": "TextGenerator",
            "description": "Generates well reasoned text for questions. Requires the full complete context.",
            "arguments": [
                {
                    "name": "prompt",
                    "type": "string",
                    "description": "Describe in detail about the question that requires an answer"
                }
            ]
        }
    ]
    </FUNCTIONS>
 
    User: """+prompt+"""Assistant: '''json'''"""
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = "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"
    )
    completion = client.chat.completions.create(
        model="mistralai/mixtral-8x22b-instruct-v0.1",
        messages=[{"role":"user","content":data}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    ret=""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            ret+=chunk.choices[0].delta.content

    print(ret)
    res = dict(eval(ret))
    print(res)
    print(type(res))

    for i in res["plan"]:
        if i['function']=="CodeGenerator" :
            memory.append({"type":"CodeGenerator","data":Code_Generator(i)})
            print(memory)
        if i['function']=="TextGenerator" :
            memory.append({"type":"TextGenerator","data":Text_Generator(i)})
            print(memory)
        if i['function']=="ImageGenerator" :
            memory.append({"type":"ImageGenerator","data":Image_Generator(i)})
            print(memory)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        main(prompt)
        for i in memory:
            if(i["type"]=="CodeGenerator"):
                st.markdown(i["data"])
                st.session_state.messages.append({"role": "assistant", "content": i["data"]})
            if(i["type"]=="TextGenerator"):
                st.markdown(i["data"])
                st.session_state.messages.append({"role": "assistant", "content": i["data"]})
            if(i["type"]=="ImageGenerator"):  
                st.markdown(st.image(i["data"]))
                st.session_state.messages.append({"role": "assistant", "content": i["data"]})
        