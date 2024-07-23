# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_plugins import MathPlugin, TextPlugin, TimePlugin
from semantic_kernel.planners import SequentialPlanner
from openai._client import AsyncOpenAI
from semantic_kernel.functions import KernelPlugin, kernel_function
from semantic_kernel.functions import KernelArguments
import os
from dotenv import load_dotenv
load_dotenv()
'''
Includes custom native plugin + custom sematic plugin with planner execution
'''
class WeatherPlugin:
    """A sample plugin that provides weather information for cities."""

    @kernel_function(name="get_weather_for_city", description="Get the weather for a city")
    def get_weather_for_city(self, city: Annotated[str, "The input city"]) -> Annotated[str, "The output is a string"]:
        if city == "Boston":
            return "61 and rainy"
        if city == "London":
            return "55 and cloudy"
        if city == "Miami":
            return "80 and sunny"
        if city == "Paris":
            return "60 and rainy"
        if city == "Tokyo":
            return "50 and sunny"
        if city == "Sydney":
            return "75 and sunny"
        if city == "Tel Aviv":
            return "80 and sunny"
        return "31 and snowing"

async def main():


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
    kernel.add_plugins({ "time": TimePlugin(), "weather" : WeatherPlugin()})
    literate_plugin = KernelPlugin.from_directory(parent_directory="plugins", plugin_name="LiterateFriend")
    kernel.add_plugin(literate_plugin)
    # create an instance of sequential planner.
    planner = SequentialPlanner(service_id=service_id, kernel=kernel)

    # the ask for which the sequential planner is going to find a relevant function.
    #ask = "What day of the week is today, all uppercase?"
    ask = "Write a short poem about the weather in Tokyo?"
    #ask = "What is the weather in Paris?"
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