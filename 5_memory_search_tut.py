# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextEmbedding
from semantic_kernel.connectors.memory.chroma.chroma_memory_store import ChromaMemoryStore
from semantic_kernel.contents import ChatHistory
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory
from openai._client import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os
from dotenv import load_dotenv
load_dotenv()
COLLECTION_NAME = "generic"


async def populate_memory(memory: SemanticTextMemory) -> None:
    # Add some documents to the ACS semantic memory
    await memory.save_information(COLLECTION_NAME, id="info1", text="My name is Andrea")
    await memory.save_information(COLLECTION_NAME, id="info2", text="I currently work as a tour guide")
    await memory.save_information(COLLECTION_NAME, id="info3", text="I've been living in Seattle since 2005")
    await memory.save_information(
        COLLECTION_NAME,
        id="info4",
        text="I visited France and Italy five times since 2015",
    )
    await memory.save_information(COLLECTION_NAME, id="info5", text="My family is from New York")


async def main() -> None:
    #kernel = Kernel()

    #azure_ai_search_settings = AzureAISearchSettings.create()
    #vector_size = 1536

    open_ai_client = AsyncOpenAI(api_key=os.environ['GROQ_API_KEY'],
                                 base_url="https://api.groq.com/openai/v1",
                                 max_retries=3)
    #import httpx
    #http_client = httpx.AsyncClient()

    kernel = Kernel()
    service_id = "my_ai_Service"
    ai_service = OpenAIChatCompletion(service_id=service_id,
                                      ai_model_id="llama3-70b-8192",
                                      # ai_model_id="gemma2-9b-it",
                                      async_client=open_ai_client,
                                      )

    kernel.add_service(ai_service)
    embedding_gen = HuggingFaceTextEmbedding(
        ai_model_id="sentence-transformers/all-MiniLM-L6-v2",
        service_id="allMiniLm",
    )
    kernel.add_service(
        embedding_gen,
    )

    acs_connector = ChromaMemoryStore(
        persist_directory="./db",
    )

    memory = SemanticTextMemory(storage=acs_connector, embeddings_generator=embedding_gen)
    kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

    print("Populating memory...")
    await populate_memory(memory)

    "It can give explicit instructions or say 'I don't know' if it does not have an answer."

    sk_prompt_rag = """
Assistant can have a conversation with you about any topic.

Here is some background information about the user that you should use to answer the question below:
{{ recall $user_input }}
User: {{$user_input}}
Assistant: """.strip()
    sk_prompt_rag_sc = """
You will get a question, background information to be used with that question and a answer that was given.
You have to answer Grounded or Ungrounded or Unclear.
Grounded if the answer is based on the background information and clearly answers the question.
Ungrounded if the answer could be true but is not based on the background information.
Unclear if the answer does not answer the question at all.
Question: {{$user_input}}
Background: {{ recall $user_input }}
Answer: {{ $input }}
Remember, just answer Grounded or Ungrounded or Unclear: """.strip()

    user_input = "Do I live in Seattle?"
    #print(f"Question: {user_input}")
    req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id="my_ai_Service")
    chat_func = kernel.add_function(
        function_name="rag", plugin_name="RagPlugin", prompt=sk_prompt_rag, prompt_execution_settings=req_settings
    )
    self_critique_func = kernel.add_function(
        function_name="self_critique_rag",
        plugin_name="RagPlugin",
        prompt=sk_prompt_rag_sc,
        prompt_execution_settings=req_settings,
    )

    chat_history = ChatHistory()
    chat_history.add_user_message(user_input)

    answer = await kernel.invoke(
        chat_func,
        user_input=user_input,
        chat_history=chat_history,
    )
    chat_history.add_assistant_message(str(answer))
    print(f"Answer: {str(answer).strip()}")
    check = await kernel.invoke(self_critique_func, user_input=answer, input=answer, chat_history=chat_history)
    print(f"The answer was {str(check).strip()}")

    print("-" * 50)
    print("   Let's pretend the answer was wrong... I lived in seattle, but ill say its Newyork now")
    print(f"Answer: {str(answer).strip()}")
    check = await kernel.invoke(
        self_critique_func, input=answer, user_input="Yes, you live in New York City.", chat_history=chat_history
    )
    print(f"The answer was {str(check).strip()}")

    print("-" * 50)
    print("   Let's pretend the answer is not related...")
    print(f"Answer: {str(answer).strip()}")
    check = await kernel.invoke(
        self_critique_func, user_input=answer, input="Yes, the earth is not flat.", chat_history=chat_history
    )
    print(f"The answer was {str(check).strip()}")

    await acs_connector.close()


if __name__ == "__main__":
    asyncio.run(main())