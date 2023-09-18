     ███████████ █████                               █████                   
    ░█░░░███░░░█░░███                               ░░███                    
    ░   ░███  ░  ░███████  ████████ ██████  ██████  ███████                  
        ░███     ░███░░███░░███░░█████░░███░░░░░███░░░███░                   
        ░███     ░███ ░███ ░███ ░░░███████  ███████  ░███                    
        ░███     ░███ ░███ ░███   ░███░░░  ███░░███  ░███ ███                
        █████    ████ ██████████  ░░██████░░████████ ░░█████                 
       ░░░░░    ░░░░ ░░░░░░░░░░    ░░░░░░  ░░░░░░░░   ░░░░░                  
                  ███████████               █████     █████                        
                 ░█░░░███░░░█              ░░███     ░░███                         
                 ░   ░███  ░████████ ██████ ░███ █████░███ █████  ██████  ████████ 
                     ░███  ░░███░░█████░░███░███░░███ ░███░░███  ███░░███░░███░░███
                     ░███   ░███ ░░░███████ ░██████░  ░██████░  ░███████  ░███ ░░░ 
                     ░███   ░███   ░███░░░  ░███░░███ ░███░░███ ░███░░░   ░███     
                     █████  █████  ░░██████ ████ █████████ █████░░██████  █████    
                    ░░░░░  ░░░░░    ░░░░░░ ░░░░ ░░░░░░░░░ ░░░░░  ░░░░░░  ░░░░░         

This repository gathers all the information about my master's thesis. This project, called `ThreatTrekker` is being
developed in collaboration with the UC3M university.

# Installation

See this [manual](./docs/instalation.md) for getting the virtual environment with all the dependencies.

## Documentation

* [Dataset Samples](./data/datasets/readme.md)
* [Useful links about possible datasets to include](resources/repos.md)

---

### How to run the script

1. Active the provided conda environment.
2. Run all the commands from the project root directory (ATTOW it is necessary due to the local imports within the
   project).
3. run python3 `main.py -h` in order to display the CLI help.

### Useful commands

* _Create a Dataframe from a dataset_

    ```bash
    python3 ./threat-hunting-ia/threat_trekker.py --build-dataset -i uwf-dataset/ -o $(date +%FT%H-%M-%S)-uwf.parquet
    ```
  > **&#9432;** The `$(date +%FT%H-%M-%S)`-uwf.parquet_ is just a fancy way of adding a timestamp to the output file.

* _Train an ML algorithm with the previously created dataframe_

    ```bash
    python3 ./threat-hunting-ia/threat_trekker.py -i uwf-dataset/parsed/2023-07-09T16-04-31-uwf.parquet  --sample 0.70
    ```
  > **&#9432;** Due to ram limitations sometime it is necessary to sample the dataset with the flag `--sample`.

---

### Aplication Flow
```mermaid
graph 
    subgraph Kafka
        T1[Topic 1]
        T2[Topic 2]
        T3[Topic 3]
    end

    subgraph Host Network
        N1[App 1]  -->|Produces| T1
        N2[App 2] -->|Procuces| T1
        N3[App N] -->|Procuces| T1
        T3 --> |Consumes| NS[SIEM]
    end

    subgraph Application
        T1 --> |Consumes| C1[App 1 Connector]

        T1 --> |Consumes| C2[App 2 Connector]
        T1 --> |Consumes| C3[App 3 Connector]
        C1 --> |Produces| T2
        C2 --> |Produces| T2
        C3 --> |Produces| T2

        T2 --> |Consumes| TT[Threat Trekker]
        TT --> |Produces| T3
    end
```
