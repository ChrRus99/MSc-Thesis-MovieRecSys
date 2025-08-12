import numpy as np
import os
import sys
import time
import tempfile
from typing import Optional, List
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pathlib import Path
from pydantic import BaseModel


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
    project_root = Path(__file__).resolve().parents[3]

# Add the project directories to the system path
sys.path.append(os.path.join(project_root, 'db_handlers'))

# Load environment variables from .env file
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)


from db_handlers.kg_rag_neo4j_db_handler import (
    get_movie_cast_and_crew_information,
)


################################################################
#   Run Instruction (from CMD): 'run service.py'               #
#   Server avaliable at web page: http://localhost:8001/docs   #
################################################################


# Create a FastAPI app
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    """Redirects the root URL to the '/docs' URL."""
    return RedirectResponse("/docs")

@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint for kg_rag_neo4j_db_handler.py
@app.get("/kg_rag_neo4j_info")
async def kg_rag_neo4j_info(entity: str, type: str, entity_id: Optional[str] = None):
    if entity_id:
        # Entity disambiguation
        return get_movie_cast_and_crew_information(entity, type, entity_id)
    else:
        # Search for the entity
        return get_movie_cast_and_crew_information(entity, type)


# @app.get("/web_search_info")
# async def web_search_info(entity: str, type: str, entity_id: Optional[str] = None):
#     # TODO
#     pass


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI server:
    #   - host="0.0.0.0" → Makes the server accessible on the local network
    #   - port=8001 → The API will be available at http://localhost:8001
    if is_docker():
        #host = "127.0.0.1"
        host = "0.0.0.0"
    else:
        host = "0.0.0.0"
    uvicorn.run(app, host=host, port=int(os.getenv("MOVIE_CAST_AND_CREW_MICROSERVICE_PORT")))