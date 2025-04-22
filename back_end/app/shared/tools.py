from typing import Any, Annotated, Literal, TypedDict, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, StructuredTool, ToolException
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from app.shared.state import InputState
from app.shared.debug_utils import tool_log


# My comment: I decided to pass state directly through make_handoff_tool function
# because if i pass state through handoff_to_agent, the state is passed indirectly by the model
# and the in the state passed by the model the messages are present correctly, while the other 
# fields of the state are resetted (which is strange). 
def make_handoff_tool(state: InputState, *, agent_name: str):
    """Create a tool that can return handoff via a Command."""
    tool_name = f"transfer_to_{agent_name}"

    #@tool(tool_name)
    def handoff_to_agent(
        # optionally pass current graph state to the tool (will be ignored by the LLM)
        #state: Annotated[InputState, InjectedState],   # state: The current state of the agent.
        # optionally pass the current tool call ID (will be ignored by the LLM)
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Redirect to another agent.

        Args:
            tool_call_id: The ID of the tool call.

        Returns:
            Command: A command to transfer to another agent.
        """
        try:
            # DEBUG LOG
            tool_log(
                function_name="handoff_to_agent: " + tool_name, 
                fields={
                    "tool_id": tool_call_id,
                    "tool_name": tool_name,
                    "transfer_to_agent": agent_name
                }
            )

            # Create a ToolMessage to signal the handoff
            tool_message = ToolMessage(
                content=f"Successfully transferred to {agent_name}.",
                name=tool_name,
                tool_call_id=tool_call_id
            )

            # Create a copy of the state and override the messages field
            updated_state = vars(state).copy()  # Copy all fields dynamically
            updated_state['messages'] = [tool_message]  # Override the messages field
            #updated_state['messages'] = updated_state.get('messages', []) + [tool_message]

            return Command(
                # navigate to another agent node in the PARENT graph
                goto=agent_name,
                graph=Command.PARENT,
                update=updated_state,
            )
        except ValueError as e:
            # Catch ValueError and rethrow as ToolException
            raise ToolException(f"An error occurred in the tool '{tool_name}': {str(e)}") from e

    # def _handle_error(error: ToolException) -> str:
    #     return (
    #         "The following errors occurred during tool execution:"
    #         + error.args[0]
    #         + "Please try again passing the correct parameters to the tool."
    #     )

    # Wrap the tool using StructuredTool for better error handling
    return StructuredTool.from_function(
        func=handoff_to_agent,
        name=f"transfer_to_{agent_name}",
        description="A tool to redirect the user to another agent.",
        #handle_tool_error=_handle_error,
    )