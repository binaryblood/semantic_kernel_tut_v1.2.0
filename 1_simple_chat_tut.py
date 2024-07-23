from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from openai._client import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
open_ai_client = AsyncOpenAI(api_key=os.environ['GROQ_API_KEY'],
             base_url="https://api.groq.com/openai/v1")
ai_service = OpenAIChatCompletion(ai_model_id="llama3-8b-8192",
                                  async_client=open_ai_client,
                                  )

kernel = Kernel()
kernel.add_service(ai_service)


response = asyncio.run(kernel.invoke_prompt("Hi!"))
print(response)