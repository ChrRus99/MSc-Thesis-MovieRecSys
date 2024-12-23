
from functools import wraps
from typing import Any, Annotated, Optional, Dict, List

from langchain_core.runnables import RunnableConfig

from main_graph.state import AgentState


ENABLE_DEBUG_LOGS = True


def generic_log(
    function_name: str, 
    messages: List[str] = None,
    fields: Dict[str, Any] = None, 
    modality: str = "normal"
) -> None:
    if ENABLE_DEBUG_LOGS:
        modality_prefix = "ERROR" if modality == "error" else "DEBUG"

        print()
        print("========================================================================================")
        print(f"Modality: {modality_prefix}")
        print(f"LOG DETAILS in function: \"{function_name}\"")
        print("========================================================================================")

        # Print messages if provided
        if messages:
            print("Prints:")
            for message in messages:
                print(f"\t{message}")
            if fields:
                print("----------------------------------------------------------------------------------------")

        # Print fields if provided
        if fields:
            print("Fields:")
            for key, value in fields.items():
                print(f"\t{key}: {value}")

        print("========================================================================================")
        print("")


def state_log(
    function_name: str, 
    state: AgentState, 
    additional_fields: Dict[str, Any] = None,
    modality: str = "normal"
) -> None:
    if ENABLE_DEBUG_LOGS:
        modality_prefix = "ERROR" if modality == "error" else "DEBUG"

        print()
        print("========================================================================================")
        print(f"Modality: {modality_prefix}")
        print(f"STATE DETAILS in node function: \"{function_name}\"")
        print("========================================================================================")
        print("Message History:")
        for message in state.messages:
            message_type = type(message).__name__
            print(f"\t{message_type} (content): {message.content}")
            #print(f"Additional Args: {message.additional_kwargs}")
            #print(f"Response Metadata: {message.response_metadata}")
            #print(f"ID: {message.id}")
        
        print("----------------------------------------------------------------------------------------")
        print("Fields:")

        # Print dynamic attributes
        for field_name in state.__dataclass_fields__.keys():
            if field_name not in ['messages']:
                field_value = getattr(state, field_name, None)
                if field_value is not None:
                    print(f"\t{field_name}: {field_value}")

        # Print additional fields if provided
        if additional_fields:
            print("----------------------------------------------------------------------------------------")
            print("Additional/External Fields:")
            for key, value in additional_fields.items():
                print(f"\t{key}: {value}")

        print("========================================================================================")
        print("")


def tool_log(
    function_name: str, 
    messages: List[str] = None,
    fields: Dict[str, Any] = None,
    modality: str = "normal"
) -> None:
    if ENABLE_DEBUG_LOGS:
        modality_prefix = "ERROR" if modality == "error" else "DEBUG"

        print()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Modality: {modality_prefix}")
        print(f"CALLED TOOL function: \"{function_name}\"")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # Print messages if provided
        if messages:
            print("Prints:")
            for message in messages:
                print(f"\t{message}")
            if fields:
                print("----------------------------------------------------------------------------------------")

        # Print fields if provided
        if fields:
            print("Fields:")
            for key, value in fields.items():
                print(f"\t{key}: {value}")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("")


def log_node_state_after_return(func):
    @wraps(func)
    async def wrapper(state: AgentState, *, config: RunnableConfig):
        # Call the function and get the result
        result = await func(state=state, config=config)  

        # Log the state after function returns
        state_log(
            function_name=func.__name__,
            state=state
        )
        
        return result

    return wrapper