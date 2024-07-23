# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_plugins import MathPlugin, TextPlugin, TimePlugin
from semantic_kernel.planners import SequentialPlanner
from openai._client import AsyncOpenAI
from semantic_kernel.functions import KernelPlugin
from semantic_kernel.functions import KernelArguments
import os
from dotenv import load_dotenv
load_dotenv()
'''
Includes example with native OOTB plugins + directory plugins executed by planner
'''
async def main():
    kernel = Kernel()

    open_ai_client = AsyncOpenAI(api_key=os.environ['GROQ_API_KEY'],
                                 base_url="https://api.groq.com/openai/v1",
                                 max_retries=3)
    kernel = Kernel()
    service_id = "my_ai_Service"
    ai_service = OpenAIChatCompletion(service_id=service_id,
                                        ai_model_id="llama3-70b-8192",
                                        #ai_model_id="gemma2-9b-it",
                                        async_client=open_ai_client,
                                      )

    kernel.add_service( ai_service)
    kernel.add_plugins({"math": MathPlugin(), "time": TimePlugin(), "text": TextPlugin()})
    literate_plugin = KernelPlugin.from_directory(parent_directory="plugins", plugin_name="LiterateFriend")
    kernel.add_plugin(literate_plugin)
    # create an instance of sequential planner.
    planner = SequentialPlanner(service_id=service_id, kernel=kernel)

    # the ask for which the sequential planner is going to find a relevant function.
    #ask = "What day of the week is today, all uppercase?"
    #ask = "Write a short poem about time travel to dinosaur age"
    ask = "What time is it now?"
    await plan_and_execute(kernel, planner, ask)

async def plan_and_execute(kernel, planner: SequentialPlanner, ask : str):
    # ask the sequential planner to identify a suitable function from the list of functions available.
    plan = await planner.create_plan(goal=ask)

    # ask the sequential planner to execute the identified function.
    result = await plan.invoke(kernel=kernel)
    print(result)

    for index, step in enumerate(plan._steps):
        print("Step:", index + 1)
        print("Description:", step.description)
        print("Function:", step._function.name)
        print("Input vars:", step._parameters)
        print("Output vars:", step._outputs)

if __name__ == "__main__":
    asyncio.run(main())