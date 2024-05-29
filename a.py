import json
from openai import OpenAI

def main(prompt):
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
    print(data)
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
    print(res["plan"][0]["arguments"])


main("Write a social media post for my ad campaign around selling more detergent. The name of the product is WishyWash, now with a new UltraClean formula, priced at $4.99. Also generate an image to go with it. Actually, while you’re at it, also add that the new product has a softner in the social media post. And brainstorm some other ideas for marketing apart from the social media post.")