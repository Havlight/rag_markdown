import argparse
import os
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import getpass
from dotenv import load_dotenv

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
根據下列資訊，回答使用者問題:

{context}

---

以中文根據上面資訊回答問題: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # if len(results) == 0 or results[0][1] < 0.1:
    #     print(f"Unable to find matching results.")
    #     return

    context_text = "\n\n---\n\n".join([doc.page_content + "\n" + doc.metadata['source'] for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"\nResponse: {response_text.content}"
    print(formatted_response)


if __name__ == "__main__":
    main()
