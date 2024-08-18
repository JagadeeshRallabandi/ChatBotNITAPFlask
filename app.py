from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# Configure the Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_college_documents():
    document_paths = [
        os.path.join("documents", "BTechRules.pdf"),
        os.path.join("documents", "HostelUndertaking.pdf"),
        os.path.join("documents", "SecurityGuidelines.pdf")
    ]
    text = ""
    for path in document_paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are the College Chatbot for the National Institute of Technology Andhra Pradesh (NIT AP). 
    As an expert on college-related matters, including academic programs, campus facilities, 
    student services, and college policies, your goal is to provide detailed and accurate responses 
    based on the provided context.

    If a question pertains to the name of the institute, respond with: 
    "The National Institute of Technology Andhra Pradesh, located in Tadepalligudem, is one of the 
    youngest NITs in India."

    For all other questions, answer as thoroughly as possible based on the provided context. 
    If the information is not explicitly mentioned in the provided context, provide a general answer 
    or direct the user to where they might find more information. Always aim to assist and guide 
    the user to the best of your ability.

    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        user_question = request.form.get("question")
        if user_question:
            answer = process_user_question(user_question)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    # Load and process documents once at startup
    documents_text = load_college_documents()
    text_chunks = get_text_chunks(documents_text)
    get_vector_store(text_chunks)
    
    app.run(debug=True)
