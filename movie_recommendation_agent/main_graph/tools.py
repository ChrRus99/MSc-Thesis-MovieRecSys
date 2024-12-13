from langchain_core.tools import tool

@tool(parse_docstring=True, response_format="content_and_artifact")
async def save_report_tool(issue: str) -> dict:
    """ A tool that saves an issue reported by the user.
        
    Args:
        issue: A brief summary of the issue reported by the user.

    Returns:
        dict: A success message and an artifact with the report data.
    """
    # Simulate saving the report
    print("Called [save_report_tool]")
    print("[TODO IMPLEMENT] Saving user issue: ", issue)
        
    # Generate the artifacts with report details
    issue_metadata = {
        "reports": [issue]
    }

    # Serialize the results
    serialized = f"Report is submitted."

    # Return content and artifact with updated state
    return serialized, issue_metadata

# TODO crea una data_class per prendere il report strutturato e salvalo come file JSON 
# config in un folder dedicato.


@tool(parse_docstring=True, response_format="content_and_artifact")
def sign_up_tool(first_name: str, last_name: str, email: str) -> dict:
    """ A tool that saves user first and last name.
        
    Args:
        first_name: The user's first name.
        last_name: The user's last name.
        email: The user's email address.

    Returns:
        dict: A success message and an artifact with the current route and user data.
    """
    # Simulate user creation
    print("Called [save_user_tool]")
    print("[TODO IMPLEMENT] Creating User:", first_name, last_name, email)

    # Generate the artifacts with user details 
    user_metadata = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email
    }

    # Serialize the results
    serialized = f"User signed up. Verification email is sent."
        
    # Return content and artifact with updated state
    return serialized, user_metadata

# TODO crea una data_class per prendere i dati strutturati dell'utente e salvali in un database.
# (insieme a film visti e ratings che verranno aggiunti successivamente)