from langchain_core.tools import tool


sign_up_prompt = f"""
You are the assistant that signs up a user. 
You need to collect user name, last name, and email and save it using tools.
"""

def sign_up_tool(return_route: str):
    """
    Creates a tool function to handle user sign-up by saving user details and returning a response.

    This factory function generates a tool that saves the user's first name, last name, and email. 
    After saving the information, it sends a verification email and provides metadata for routing.

    Args:
        return_route (dict): The route to navigate to after the user signs up.

    Returns:
        Callable: A tool function that accepts user details and returns a success message along 
                  with updated routing and user data as a dictionary.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(first_name: str, last_name: str, email: str) -> dict:
        """ A tool that saves user first and last name.
        
        Args:
            first_name: The user's first name.
            last_name: The user's last name.
            email: The user's email address.

        Returns:
            dict: A success message and an artifact with the current route and user data.
        """
        # Simulate user creation
        print("Creating User [save_user_tool]:", first_name, last_name, email)

        # Generate the artifact with routing information and user details 
        user_metadata = {
            "current_route": return_route,
            "first_name": first_name,
            "last_name": last_name,
            "email": email
        }

        serialized = f"User signed up. Verification email is sent."
        
        # Return content and artifact with updated state
        return serialized, user_metadata
    
    return tool_func