from typing import List
from PIL import Image
import streamlit as st
import time
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient
import os
import io
import logging


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import torch

DIMENSION=2048

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

milvus_db = "milvus_demo_5.db"
milvus_collection = milvus_db.split(".")[0]

if "client" not in st.session_state:
    st.session_state.client = MilvusClient(milvus_db)

if "ef" not in st.session_state:
    st.session_state["ef"] = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = st.session_state["ef"].dim["dense"]

if "model" not in st.session_state:
    # Load the embedding model with the last layer removed
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    st.session_state["model"] = model
    from torchvision import transforms

    # Preprocessing for images
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    st.session_state["preprocess_image"] = preprocess

def dense_search(col, query_dense_embedding, limit=5):
    client = st.session_state.client
    search_params = {
        "metric_type": "IP",
        "params": {
            "radius": 0.6,
        }}

    res = client.search(
        collection_name=milvus_collection,
        data=[query_dense_embedding],
        anns_field=col,
        limit=limit,
        output_fields=["text","local_image","country","author_name"],
        search_params=search_params,
    )
    
    return res[0]

def dense_image_search(col, query_dense_embedding, limit=5):
    client = st.session_state.client
    search_params = {
        "metric_type": "L2",
        "params": {
             "radius": 300,
        }
        }

    res = client.search(
        collection_name=milvus_collection,
        data=[query_dense_embedding],
        anns_field=col,
        limit=limit,
        output_fields=["text","local_image","country","author_name"],
        search_params=search_params,
    )
    
    return res[0]

def generate_multiple_questions(question: str):
    
    messages = [
        SystemMessage(
            content=f"You are a helpful assistant specialized in rewording questions provided by a user, to maximize the retrieval capabilities of the query.\n\n Given the user prompt, proppose 5 new prompts, which should be explicit, each prompt must be different and use a different point of view, prompts should be optimized for retrival and prompts will be shown as bullet points. The output must be only the bullet points\n\nUser prompt: {question}"
        )
    ]

    llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0)
    response = llm.invoke(messages)
    result = response.content
    
    reworded_questions = result.strip().split("-")

    return reworded_questions

def rerank_results(results):

    results_dict = {}
    for result in results:
        text = result['entity']['text']
        if text not in results_dict:
            results_dict[text] = result
            results_dict[text]['count']=1
        else:
            results_dict[text]['count']+=1
    
    reranked_results = list(results_dict.values())
    for result in reranked_results:
        result['distance'] = result['distance']/result['count']
    
    logging.info("Results after reranking")
    print(results_dict)

    return reranked_results

def search(prompts: List[str]):
    # Enter your search query
    prompts = [x for x in prompts if len(x) > 5]

    # Generate embeddings for the query
    query_embeddings = st.session_state["ef"](prompts)
    
    dense_results = []
    for r in query_embeddings:
        dense_results += dense_search("dense_vector", query_embeddings["dense"][0])

    reranked_results = rerank_results(dense_results)

    return reranked_results


def generate_final_answer(context: str, prompt) -> str:

    response = context

    messages = [
        SystemMessage(
            content="You are a helpful assistant specialized in fast fact checking! Given the provided context:\n\n1) you have to answer if the user query is True, False or Partially False \n\n 2) After providing the result, explain your reasoning step by step"
        ),
        HumanMessage(
            content=f"Context: \n\n {context} \n\n User question: {prompt}"
        )
    ]

    llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0)
    response = llm.invoke(messages)
    result = response.content
    return context + "\n\n# Final Response\n\n"+result.strip()

def embed_image(data):
    with torch.no_grad():
        ret = st.session_state["model"](torch.stack(data))
        # If more than one image, use squeeze
        if len(ret) > 1:
            return ret.squeeze().tolist()[0]
        # Squeeze would remove batch for single image, so using flatten
        else:
            return torch.flatten(ret, start_dim=1).tolist()[0]

# Streamed response emulator

def format_response(text):

    text = text.replace("reviewed claim:", "**reviewed claim**:")
    text = text.replace("unverified claim:", "\n\n**unverified claim**:")
    text = text.replace("Both claims ", "\n\nBoth claims ")

    return text

def response_generator():
    if len(st.session_state.messages) > 1: # user input
        # user input and image
        if uploaded_file is not None:
            user_prompt = st.session_state.messages[-1]['content']
            logging.info(f"El usuario ha escrito el prompt: {user_prompt}")
            prompts = generate_multiple_questions(user_prompt)
            logging.info(f"Generated the following prompts with AI: {str(prompts)}")

            queries_text = "# Queries proposed by the system:\n\n"
            queries_text += "- ".join(prompts) + "\n\n"

            logging.info(f"Quering database")
            results = search(prompts)

            bytes_data = uploaded_file.getvalue()
            image_data = image = Image.open(io.BytesIO(bytes_data))
            
            preprocessed_image = st.session_state["preprocess_image"](image_data)
            img_embedding = embed_image([preprocessed_image])
            images_results = dense_image_search('image_vector',img_embedding)

            results = rerank_results(results+images_results)

            logging.info(f"Found {len(results)} results")
            if len(results) < 1: 
                logging.info(f"Not results found inside database")

                response = "I don't have information related to that claim in my database, so i can't provide a response"
            else:
                print("===============")
                print(results)
                
                response = "\n\n # Claims"
                for i in range(0, len(results)):
                    response += f"""\n\n## Claim {i+1} - {results[i]['entity']['author_name']} ({results[i]['entity']['country']})\n\n {results[i]['entity']['text']}"""
                    response = format_response(response)
                    
                    if len(results[i]['entity']['local_image']) > 0:
                        image = results[i]['entity']['local_image']
                        image = image.replace("cr_images","http://localhost:8000")
                        response += f"![image]({image}) \n\n"

                
                response = queries_text + generate_final_answer(response, user_prompt)

        else: # only user input
            user_prompt = st.session_state.messages[-1]['content']
            logging.info(f"El usuario ha escrito el prompt: {user_prompt}")
            prompts = generate_multiple_questions(user_prompt)
            logging.info(f"Generated the following prompts with AI: {str(prompts)}")

            queries_text = "# Queries proposed by the system:\n\n"
            queries_text += "- ".join(prompts) + "\n\n"

            print(queries_text)

            logging.info(f"Quering database")
            results = search(prompts)
            logging.info(f"Found {len(results)} results")
            if len(results) < 1: 
                logging.info(f"Not results found inside database")

                response = "I don't have information related to that claim in my database, so i can't provide a response"
            else:
                print("===============")
                print(results)
                
                response = "\n\n # Claims"
                for i in range(0, len(results)):
                    response += f"""\n\n## Claim {i+1} - {results[i]['entity']['author_name']} ({results[i]['entity']['country']})\n\n {results[i]['entity']['text']}"""
                    response = format_response(response)
                    
                    if len(results[i]['entity']['local_image']) > 0:
                        image = results[i]['entity']['local_image']
                        image = image.replace("cr_images","http://localhost:8000")
                        response += f"![image]({image}) \n\n"

                
                response = queries_text + generate_final_answer(response, user_prompt)

    #no user input
    else: # only image
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            image_data = image = Image.open(io.BytesIO(bytes_data))
            
            preprocessed_image = st.session_state["preprocess_image"](image_data)
            img_embedding = embed_image([preprocessed_image])
            images_results = dense_image_search('image_vector',img_embedding)
            logging.info("Image results")
            logging.info(images_results)
            response = 'Image ready to be used!'
            
            images_results = rerank_results(images_results)

            for result in images_results:
                result['entity']['local_image'] = result['entity']['local_image'].replace("cr_images","http://localhost:8000")
                result['entity']['text'] = format_response(result['entity']['text'])

            response = '\n\n'.join([f"### {result['entity']['author_name']} ({result['entity']['country']}) \n\n![image]({result['entity']['local_image']}) \n\n{result['entity']['text']}\n\n" for result in images_results])
        else:
            response = "What do you want to check?"
    
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)

st.title(":ocean: FISTERRA	:octopus:")
st.markdown("(**F**iltro **I**nteligente y **S**istemático de **T**extos **E**rróneos y **R**umores **R**ápidamente **A**nalizados")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Choose a JPG file", accept_multiple_files=False,
        type=['png', 'jpg', 'jpeg']
    )


# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})