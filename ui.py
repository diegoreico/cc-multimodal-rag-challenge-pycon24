from typing import List
import streamlit as st
import time
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient
import os


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

milvus_db = "milvus_demo_5.db"
milvus_collection = milvus_db.split(".")[0]

if "client" not in st.session_state:
    st.session_state.client = MilvusClient(milvus_db)

if "ef" not in st.session_state:
    st.session_state["ef"] = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = st.session_state["ef"].dim["dense"]

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
    reranked_results = list(results_dict.values())
    return reranked_results

def search(prompts: List[str]):
    # Enter your search query
    prompts = [x for x in prompts if len(x) > 5]
    print(prompts)

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
    return context + "\n\n**Final Response**:\n\n"+result.strip()

# Streamed response emulator
def response_generator():
    if len(st.session_state.messages) > 1:
            
            user_prompt = st.session_state.messages[-1]['content']
            prompts = generate_multiple_questions(user_prompt)

            queries_text = "Queries proposed by the system:\n\n"
            queries_text += "- ".join(prompts) + "\n\n"

            print(queries_text)

            results = search(prompts)
            if len(results) < 1: 
                response = "I don't have information related to that claim in my database, so i can't provide a response"
            else:
                print("===============")
                print(results)
                
                response = ""
                for i in range(0, len(results)):
                    response += f"""**Claim {i+1}** - {results[i]['entity']['author_name']} ({results[i]['entity']['country']})\n\n {results[i]['entity']['text']}"""
                    response = response.replace("reviewed claim:", "**reviewed claim**:")
                    response = response.replace("unverified claim:", "\n\n**unverified claim**:")
                    response = response.replace("Both claims ", "\n\nBoth claims ")
                    if len(results[i]['entity']['local_image']) > 0:
                        image = results[i]['entity']['local_image']
                        image = image.replace("cr_images","http://localhost:8000")
                        response += f"![image]({image}) \n\n"

                response+="\n\nGenerating a result...\n\n"

                
                response = queries_text + generate_final_answer(response, user_prompt)

    else:
        response = "Que quieres que compruebe?"
    
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)

st.title("The Mother Fact Checker")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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