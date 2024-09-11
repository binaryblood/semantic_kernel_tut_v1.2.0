# Copyright (c) Microsoft. All rights reserved.
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planners.function_calling_stepwise_planner import FunctionCallingStepwisePlanner


async def main():
    # <CreatePlanner>
    # Initialize the kernel
    open_ai_client = AsyncOpenAI(api_key=os.environ['GROQ_API_KEY'],
                                 base_url="https://api.groq.com/openai/v1",
                                 max_retries=3)
    # import httpx
    # http_client = httpx.AsyncClient()

    kernel = Kernel()
    service_id = "default"
    ai_service = OpenAIChatCompletion(service_id=service_id,
                                      ai_model_id="llama3-70b-8192",
                                      # ai_model_id="gemma2-9b-it",
                                      async_client=open_ai_client,
                                      )

    kernel.add_service(ai_service)

    script_directory = os.path.dirname(__file__)
    plugins_directory = os.path.join(script_directory, "plugins")
    kernel.add_plugin(parent_directory=plugins_directory, plugin_name="MathPlugin")

    planner = FunctionCallingStepwisePlanner(service_id="default")
    # </CreatePlanner>
    # <RunPlanner>
    goal = "Figure out how much I have if first, my investment of 2130.23 dollars increased by 23%, and then I spend $5 on a coffee"  # noqa: E501
    print(planner)
    print("=="*30)
    # Execute the plan
    result = await planner.invoke(kernel=kernel, question=goal)
    print(result)
    print("==" * 30)
    print(f"The goal: {goal}")
    print(f"Plan result: {result.final_answer}")
    # </RunPlanner>


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())