from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from metaphor_python import Metaphor
from alive_progress import alive_bar
import arxiv
import json
import os
import openai
import tiktoken

# openai.api_key = os.getenv("OPENAI_API_KEY")
# metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))

openai.api_key = "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"
metaphor = Metaphor("ce19645b-9b03-4321-bd44-31050f7e1616")

with open("prompts.json", 'r') as json_file:
    prompts = json.load(json_file)

def conversational_qa(arxiv_id: str):
    """Conversational question answering on a given ArXiv paper."""
    with alive_bar(0, title="Loading the paper into memory...") as bar:
        text = load_arxiv(arxiv_id)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 50,
        )
        documents = text_splitter.create_documents([text])

        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai.api_key)
        vectorstore = Chroma.from_documents(documents, embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(
                OpenAI(
                    temperature=0, 
                    openai_api_key=openai.api_key,
                    streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()]
                ),
            vectorstore.as_retriever(), 
            memory=memory,
        )
        bar()

    # Streaming callback handler is messing with the I/O
    i = 0
    print(f"\nCHAT: {prompts['begin_chat']}\n{prompts['commands']}\n")
    while True:
        if i > 0:
            print()
        user_input = input("\nUSER: ")
        if user_input == "q" or user_input == 's':
            print()
            return user_input
        print()
        result = qa({"question": user_input})['answer']
        i += 1
        
    

def search_arxiv(query: str, start_published_date=None):
    """Searches ArXiv for papers related to a given query and provides summarized information on the top result."""
    # metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))

    search_response = metaphor.search(
        query=f"Here is a paper on {query}",
        use_autoprompt=False,
        # start_published_date="2023-06-01",
        include_domains=["https://arxiv.org/"],
        num_results=5,
    )

    arxiv_ids = [result.url.split("/")[-1][:10] for result in search_response.results]
    arxiv_ids = list(set(arxiv_ids))

    summarized = {}

    # Summarizing the abstract of the papers Metaphor returned
    print(f"\nHere are the top {len(arxiv_ids)} most relevant papers I found on '{query}':\n")
    for i, arxiv_id in enumerate(arxiv_ids):
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        print(f"{i+1}. {paper.title}")
        print(f"Publication date: {paper.published.strftime('%Y-%m-%d')}")

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompts["summarize_abstract"]},
                {"role": "user", "content": f"Abstract:\n {paper.summary}"},
            ],
            stream=True,
        )
        collected_response = ""
        for chunk in completion:
            chunk_message = chunk['choices'][0]['delta'].get('content', '')
            print(chunk_message, end="")
            collected_response += chunk_message

        summarized[arxiv_id] = {
            "title": paper.title,
            "abstract": paper.summary,
            "summary": collected_response,
        }
        print('\n')
    
    # Asking ChatGPT to output most relevant paper
    user_message = f"User query: {query}\n\n"
    for i, arxiv_id in enumerate(summarized.keys()):
        user_message += f"{i+1}. Title: {summarized[arxiv_id]['title']}\n\n"
        user_message += f"{summarized[arxiv_id]['summary']}\n\n"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompts["rank_papers"]},
            {"role": "user", "content": user_message},
        ],
    )

    rank = completion.choices[0].message.content
    # Extract recommended number
    for c in rank:
        if c.isdigit():
            rank = int(c) - 1
            break
    if rank > 4 or isinstance(rank, str):
        rank = 0
    arxiv_id = arxiv_ids[rank]

    print(f"\nChatGPT recommended paper as most relevant: {rank+1}. {summarized[arxiv_id]['title']}")
    return summarized, rank


def load_arxiv(arxiv_id: str):
    """Load and extract text content from a specific ArXiv paper using its unique ArXiv identifier."""
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF package not found, please install it with "
            "`pip install pymupdf`"
        )

    search = arxiv.Search(id_list=[arxiv_id])
    result = next(search.results())

    doc_file_name = result.download_pdf()
    with fitz.open(doc_file_name) as doc_file:
        text: str = "".join(page.get_text() for page in doc_file)
    os.remove(doc_file_name)

    return text.split("References\n")[0]

def num_tokens_from_string(string: str, encoding_name: str='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
