import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pathlib import Path

def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

# Add the project directories to the system path
if is_docker():
    print("[INFO] Running inside a Docker container")

    # Set the project root
    project_root = "/app"
else:
    print("[INFO] Running locally")

    # Dynamically find the project root
    project_root = Path(__file__).resolve().parents[2]

# Add the project directories to the system path
sys.path.append(os.path.join(project_root, 'back_end'))

# Load environment variables from .env file
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

from app.app_agent import create_app_agent



################################################################
#   Run Instruction (from CMD): 'run service.py'               #
#   Server avaliable at web page: http://localhost:8000        #
################################################################


app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, create_app_agent, path="/movie-agent")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)