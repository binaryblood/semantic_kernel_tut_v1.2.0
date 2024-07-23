from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from openai._client import AsyncOpenAI
from semantic_kernel.functions import KernelPlugin
from semantic_kernel.functions import KernelArguments
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
open_ai_client = AsyncOpenAI(api_key=os.environ['GROQ_API_KEY'],
             base_url="https://api.groq.com/openai/v1")
kernel = Kernel()
ai_service = OpenAIChatCompletion(ai_model_id="llama3-8b-8192",
                                  async_client=open_ai_client,
                                  )




kernel.add_service(ai_service)

literate_plugin = KernelPlugin.from_directory(parent_directory="plugins", plugin_name="LiterateFriend")
kernel.add_plugin(literate_plugin)

arguments = KernelArguments(topic="time travel to dinosaur age")

result = asyncio.run( kernel.invoke(literate_plugin["ShortPoem"], arguments))
print(result)