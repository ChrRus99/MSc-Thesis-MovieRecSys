# Instructions for running The Movie Recommendation System


## Step 1: Start databases
* (Optional) Start **Postgres SQL** (application): only in case of connection errors, it should not be necessary.

* (Optional) Start **MongoDB** (application): only in case of connection errors, it should not be necessary.

* Start **MinIO** (service): 
    ```CMD
    cd C:\Program Files\MinIO
    minio.exe server --address :9500 --console-address :9501 D:\MinIO\data
    ```

    > **Note:** With these commands we are:
    > - Running `minio.exe` from the directory where it is stored (`C:\Program Files\MinIO`)
    > - Saving MinIO `data` in the directory `D:\MinIO\data`.
    > - Running MinIO on custom port `9500` and its web console on custom port `9501`.

    > **Note:** Default command: `minio.exe server D:\MinIO\data`.

* (Optional) Open **MinIO web UI** (browser): at [http://192.168.1.29:9501/browser](http://192.168.1.29:9501/browser).
    > **Note:** see prompt output for sign-in credentials. 

* Start **Hadoop**, **HBase** and **Kafka** (services): run (as administrator) `start_hadoop_hbase_kafka.bat` executable located at path `D:\Internship\recsys\`.
     > **Important Note:** This executable must be run as **administrator**!

* (Optional) Start **Neo4j** (application): only in case of connection errors, it should not be necessary.


## Step 2: Start Airflow data pipelines

* Start **Docker Desktop** (application).

* Start **Airflow web UI** (browser): to do that compile and run docker-compose.yaml:
    ```CMD
    cd D:\Internship\recsys
    docker compose up -d
    ```
    > **Note:** Other useful commands:
    > - `docker compose down`: Stops and removes containers, networks, volumes, and images created by `docker compose up`:
    >   ```CMD
    >   docker compose down -v
    >   ```
    > - `docker ps`: Lists all running Docker containers.
    >   ```CMD
    >   docker ps
    >   ```
    > - `docker exec -it <container_name> bash`: Opens an interactive shell inside a running container (optionally use `/bin/bash` to specify the shell explicitly). 
    >   ```CMD
    >   docker exec -it <container_name> /bin/bash
    >   ```
    > - `docker logs <container_name>`: Displays logs for a specific container.

* Open **Airflow web UI** (browser): at [http://localhost:8080](http://localhost:8080).

## Step 3: Start event handlers
* Start **mongodb event listener and kafka producer** `event_producers/mongodb_event_listener_and_kafka_producer.py` (service):
    ```CMD
    cd D:\Internship\recsys\event_handlers\event_producers\mongodb_event_listener_and_kafka_producer.py
    python mongodb_event_listener_and_kafka_producer.py
    ```

* Start **kafka consumer and airflow triggerer** `event_consumers/kafka_consumer_and_airflow_triggerer.py` (service):
    ```CMD
    cd D:\Internship\recsys\event_handlers\event_consumers\kafka_consumer_and_airflow_triggerer.py
    python kafka_consumer_and_airflow_triggerer.py
    ```

* Start all **data pipelines** from Airflow web UI.

## Step 4: Start microservices
* Start **movie cast and crew info microservice** `movie_cast_and_crew_info/service.py` (service):
    ```CMD
    cd D:\Internship\recsys\microservices\movie_cast_and_crew_info
    python service.py
    ```
    > **Note:** API web UI will be available at [http://localhost:8001](http://localhost:8001)

* Start **movie recommendation microservice** `movie_recommendation/service.py` (service):
    ```CMD
    cd D:\Internship\recsys\microservices\movie_recommendation
    python service.py
    ```
    > **Note:** API web UI will be available at [http://localhost:8002](http://localhost:8002)

* Start **user info microservice** `user_info/service.py` (service):
    ```CMD
    cd D:\Internship\recsys\microservices\user_info
    python service.py
    ```
    > **Note:** API web UI will be available at [http://localhost:8003](http://localhost:8003)


## Step 5: Start back-end
* Start **back-end** (service):
    ```CMD
    cd D:\Internship\recsys\back_end\app
    fastapi run server.py
    ```
    > **Note:** API web UI will be available at [http://localhost:8000](http://localhost:8000) 

## Step 6: Start front-end
* Start **front-end** (service):
    ```CMD
    cd D:\Internship\recsys\front_end
    streamlit run chat_ui_client.py
    ```
    > **Note:** API web UI will be available at [http://localhost:8501](http://localhost:8501) 