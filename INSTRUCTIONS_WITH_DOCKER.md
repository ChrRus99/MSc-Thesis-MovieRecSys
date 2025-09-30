# Instructions for running The Movie Recommendation System using Docker (docker-compose)


## Step 1: Run docker-compose
* Start **Docker Desktop** (application).

* (Optional) Reset Docker containers, networks, and the default volumes (if built before):
    ```CMD
    cd D:\Internship\recsys
    docker-compose down -v
    ```
    > **Note:** -v is used to delete volumes as well.

* Build and run Docker containers, networks, and the default volumes:
    ```CMD
    cd D:\Internship\recsys
    docker-compose up -d
    ```
    > **Note:** -d to run in detached mode (background).

> **Note:** Other useful commands:
> - `docker ps`: Lists all running Docker containers.
>   ```CMD
>   docker ps
>   ```
> - `docker exec -it <container_name> bash`: Opens an interactive shell inside a running container (optionally use `/bin/bash` to specify the shell explicitly). 
>   ```CMD
>   docker exec -it <container_name> /bin/bash
>   ```
> - `docker logs <container_name>`: Displays logs for a specific container.


## Step 2: Start Airflow data pipelines
* Open **Airflow web UI** (browser): at [http://localhost:8080](http://localhost:8080).

* Start the 4 pipelines from the Airflow Web UI:
    - Activate/Unpause the 4 pipelines from the Airflow Web UI
    - Manually run `init_pipeline` to initialize the program AND WAIT UNTIL IT TERMINATES RUNNING!
    - (Optional) Run the `offline_pipeline`.

    > **Note:** 
    > - You do not need to run the `offline_pipeline` and the `periodic_update_pipeline` manually, they are triggered automatically by the program.
    > - You do not need to run the `online_pipeline` manually, it is triggered automatically by the program.


## Step 3: Start back-end
* Start **back-end** (service):
    ```CMD
    cd D:\Internship\recsys\back_end\app
    fastapi run server.py
    ```
    > **Note:** API web UI will be available at [http://localhost:8000](http://localhost:8000) 

## Step 4: Start front-end
* Start **front-end** (service):
    ```CMD
    cd D:\Internship\recsys\front_end
    streamlit run client.py
    ```
    > **Note:** API web UI will be available at [http://localhost:8501](http://localhost:8501) 