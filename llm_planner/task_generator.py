def get_llm_response(client, prompt, model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content

from groq import Groq
client = Groq(api_key="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")

prompt = ""
model = "llama-3.1-70b-versatile"
output_file = "test.txt"
ctr = 0

while True:
    try:
        response = get_llm_response(client, prompt, model)
        with open(output_file, "a") as file:
            file.write(f"TASK-{str(ctr).zfill(5)}----------------- \n")
            for line in response:
                file.write(line)
            file.write("--------------------------------------- \n")
            print("Response saved to txt file.")
        ctr+=1
    except:
        raise BufferError; "Error in getting response from groq"