import torch

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer , AutoModelForCausalLM , pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

Loader = PyPDFLoader
FILE_PATH = "/home/nghiavm1/workdir/chatbot/triethoc.pdf"
loader = Loader(FILE_PATH)
documents = loader.load ()

text_splitter = RecursiveCharacterTextSplitter(chunk_size =100, chunk_overlap =20)

docs = text_splitter.split_documents(documents)

print("Number of sub - documents : ", len(docs))
# print(docs[0])

embedding = HuggingFaceEmbeddings()

vector_db = Chroma.from_documents(documents =docs, embedding = embedding)
retriever = vector_db.as_retriever ()

result = retriever.invoke ("Projects name?")
# print("Number of relevant documents : ", len(result))

nf4_config = BitsAndBytesConfig(
  load_in_4bit = True ,
  bnb_4bit_quant_type = "nf4",
  bnb_4bit_use_double_quant = True ,
  bnb_4bit_compute_dtype = torch.bfloat16
)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME,
  quantization_config = nf4_config,
  low_cpu_mem_usage = True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model_pipeline = pipeline(
  "text-generation",
  model = model,
  tokenizer = tokenizer,
  max_new_tokens =512,
  pad_token_id = tokenizer.eos_token_id,
  device_map = "auto"
)

llm = HuggingFacePipeline (
  pipeline = model_pipeline,
)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
  {"context": retriever | format_docs , "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

USER_QUESTION = "What is my interests?"
output = rag_chain.invoke(USER_QUESTION)
if output.find("Answer: ") >= 0:
    answer = output.split("Answer: ")
else:
   answer = output
print(answer)