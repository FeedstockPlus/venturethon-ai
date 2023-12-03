import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import InvalidRequestError
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import re

from openai.error import AuthenticationError
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import os

import logging

# Load environment variables from .env
load_dotenv()


def prompt_template():

    _template = """ You are chatbot Assistant

    Instructions for the chatbot:

    In general, you should inform the users that compliance needs to be confirmed with an accredited environmental auditor. you can give guidance and recommendations but can not always guarantee accurate because compliance depends on many factors.
    
    You is especially trained for European environmental regulation when it comes to bioenergy, biofuels and Renewable Energy Directive II. The Chatbot should inform the user about this when asked about regulation/compliance situations outside of Europe.
    
    You should make reference as much as possible to the Renewable Energy Directive II. When making such reference, You could also inform about possible changes to be expected in the future versions of the Directive, when the Renewable Energy Directive III will be implemented.
    
    The Renewable Energy Directive is at the European level. EU Member States had until 1 July 2021 to transpose REDII into national law. Sometimes, EU Member States make further limitations or rules to the applicability or eligibility of feedstocks, or add more criteria to the sustainable criteria in place.
    
    When asked about ways to sell Feedstocks or connect to buyers, You could inform the users of general possibilities such as connecting to buyers during events, industry conferences and B2B matchmaking opportunities, or also connect through industry bioenergy associations. The chatbot should also inform the user that the FeedstockPlus team is currently developing a dedicate solutions: “our team is currently developing a platform called FeedstockPlus Source, which will allow Feedstocks Suppliers and Buyers as well as the whole stakeholders involved to Connect, Transat, and ensure Regulatory Compliance especially around bioenergy, biofuels, the impact of the Renewable Energy Directive in the EU as well as other relevant European and national law concerning bioenergy, biomass, waste material and circular economy”. You should be engaging with the user and ask questions such as “do you want to sign up for the platform? Do you want to try the platform? Would such a platform be relevant for you?
    
    Users may ask about carbon emissions, GHG emissions, Green House Gas emissions, pollution, Carbon intensity and other similar terms related to feedstocks. You should mainly refer to the Renewable Energy Directive when it comes to such information and data point. The Directive and its annexes, sometimes complemented by national application restrictions, detail how to calculate exactly GHG emissions from specific types of Feedstocks.
    
    The user will ask questions to You about feedstocks. In general, You should ask as many questions as possible to the user in order to understand what feedstock he has: the type of feedstocks, the country of origin, the quantity or volume, how frequently these feedstocks are available, if there are specific certifications or labels attached to these feedstocks. If the user has started to give information about his specific feedstocks, then You should always refer specifically to this feedstock to give further answers in the conversation.
    
    When asked how a Feedstock suppliers can be REDII certified, You should give proper answers. REDII certification is delivered by demonstrating compliance to one of the voluntary certification schemes that are approved by the European Commission such as RSB, ISCC, REDcert, SBP, 2bsvs, etc.

    
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    Memory_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    return Memory_QUESTION_PROMPT


def ask_question(question, qa, chat_history):
    try:
        result = qa({"question": question, "chat_history": chat_history})
        
        chat_history.append((question, result['answer']))
        st.write(f"-> **Question**: {question}")
        st.write(f"**Answer**: {result['answer']}")
        
        st.button()
    except InvalidRequestError:
        st.write("Try another chain type as the token size is larger for this chain type")


def display_chat_history(chat_history):
    
    for i, (question, answer) in enumerate(chat_history):
        st.info(f"Question {i + 1}: {question}")
        st.success(f"Answer {i + 1}: {answer}")
        st.write("----------")


def load_embeddings(api_key):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local("embeedingRedgpt/global_embeeding", embedding)
    return embedding, db


def process_uploaded_files(uploaded_files,api_key):
    embedding, db = load_embeddings(api_key)

    loader = PyPDFDirectoryLoader("embeedingRedgpt/useruploaded")

    docs = loader.load_and_split(text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        ))
    metadata = []  # Empty list to store metadata
    pages=[]
    for i in range(len(docs)):
        # print(i)
        # print(docs[i].metadata)
        metadata.append(docs[i].metadata)
        pages.append(docs[i].page_content)

    db =  FAISS.from_texts(pages, embedding,metadatas=metadata)
    save_path="embeedingRedgpt/"
    db.save_local(save_path)

    db_global = FAISS.load_local("embeedingRedgpt/global_embeeding", embedding)
    db_global.merge_from(db)
    db_global.save_local("embeedingRedgpt/global_embeeding")

    return db




# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["history"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.clear()
    # st.session_state.memory.buffer.clear()
    # st.session_state.memory


def main():

    st.title("Ask Your Feedstock Chatbot ")
     
    api_key=os.getenv("openAi_key")


    if api_key:
            
        try    :
        
            embedding, db = load_embeddings(api_key)
            

            with st.sidebar.expander("File Uploader") :

                uploaded_files = st.file_uploader(
                    "Choose PDF files", type=["pdf"], accept_multiple_files=True
                )

            if uploaded_files:
                    
                # Create a folder to save the uploaded files
                folder_name = "dbuseruploaded"
                os.makedirs(folder_name, exist_ok=True)

                # Save the uploaded files to the folder
                for file in uploaded_files:
                    file_path = os.path.join(folder_name, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                db = process_uploaded_files(uploaded_files,api_key)

            retriever = db.as_retriever()
            model = OpenAI(temperature=0, openai_api_key=api_key)

            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Hello ! Ask me anything about Feedstock 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey ! 👋"]

            if 'memory' not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                    )

            get_prompt=prompt_template()
            question_generator = LLMChain(llm=model, prompt=get_prompt)
        
            doc_chain = load_qa_with_sources_chain(llm=model, chain_type="stuff")

            chain = ConversationalRetrievalChain(
                retriever=retriever,
                memory=st.session_state.memory,
                
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
            )

                #container for the chat history
            response_container = st.container()
            #container for the user's text input
            container = st.container()
            

            def conversational_chat(query):
                try:
            
                    with get_openai_callback() as cb:
                        result = chain({"question": query, "chat_history": st.session_state['history']})
                        reference_Doc = db.similarity_search(query)
                        answer = re.sub(r'SOURCES:.*', '', result["answer"])
                        

                    
                        st.session_state['history'].append((query, answer))
                    
                        return answer,reference_Doc
                    
                
                except InvalidRequestError:
                    st.write("Try another chain type as the token size is larger for this chain type")
            

            # Allow to download as well
            download_str = []
            query=""
            with container:
                
                with st.form(key='my_form', clear_on_submit=True):
                    
                    user_input = st.text_input("Query:", placeholder="Type Your Query (:", key='input')
                    submit_button = st.form_submit_button(label='Send',type='primary')


                if submit_button and user_input:
                    output,reference_Doc = conversational_chat(user_input)
                    output, reference_Doc = conversational_chat(user_input)
                    print(output)
                    if output.strip() == "I don't know.":
                        output = "I'm still learning and don't have the information you're looking for."
                    print(output)
                    
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)
                    
                    
                    if st.session_state['generated']:
                        with response_container:
                            for i in range(len(st.session_state['generated'])):
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
                                download_str.append(st.session_state["past"][i])
                                download_str.append(st.session_state["generated"][i])
                    
                    with st.expander("Show Reference Documents:"):
                        refercemetadata = []  # Empty list to store metadata
                        refercepages=[]
                        for i in range(len(reference_Doc)):

                            refercemetadata.append(reference_Doc[i].metadata)
                            refercepages.append(reference_Doc[i].page_content)

                        st.write(refercemetadata)
                
                
                # download_str = '\n'.join(download_str)
                # if download_str:
                #         st.sidebar.download_button('Download Conversion',download_str)

                # with st.sidebar.expander("**View Chat History**"):
                #     display_chat_history(st.session_state.history)


                if st.session_state.history:   
                    if st.button("Clear-all",help="Clear all chat"):
                        st.session_state.history=[]

                # st.session_state.me
                st.button("New Chat", on_click = new_chat, type='primary')


        except AuthenticationError as e :
            link = "[Click here](https://platform.openai.com/account/api-keys)"
            st.error(f"Ensure the API key used is correct, clear your browser cache, or generate a new one {link}")   


    else :
        st.sidebar.warning("Please Enter Api Key in env")    




if __name__ == '__main__':
    main()
