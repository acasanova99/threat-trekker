# Datasets are not included in the git project due to its size.

## List of datasets

* [UWF-DATASET-2022](https://datasets.uwf.edu/)

### Parsed Data samples

* UWF-DATASET-2022:

```commandline
 #   Column          Dtype         
---  ------          -----         
 0   resp_pkts       int32         
 1   service         object        
 2   orig_ip_bytes   int32         
 3   local_resp      bool          
 4   missed_bytes    int32         
 5   proto           object        
 6   duration        float64       
 7   conn_state      object        
 8   dest_ip_zeek    object        
 9   orig_pkts       int32         
 10  community_id    object        
 11  resp_ip_bytes   int32         
 12  dest_port_zeek  int32         
 13  orig_bytes      float64       
 14  local_orig      bool          
 15  datetime        datetime64[ns]
 16  history         object        
 17  resp_bytes      float64       
 18  uid             object        
 19  src_port_zeek   int32         
 20  ts              float64       
 21  src_ip_zeek     object        
 22  label_tactic    object
 
         resp_pkts service  ...  src_ip_zeek    label_tactic
0                4    None  ...  143.88.7.10       Discovery
1                4    None  ...  143.88.7.10       Discovery
2                1    None  ...  143.88.7.10       Discovery
3                1    None  ...  143.88.7.10       Discovery
4                1    None  ...  143.88.7.10       Discovery
...            ...     ...  ...          ...             ...
9280801         44    None  ...  143.88.2.10  Reconnaissance
9280802         44    None  ...  143.88.2.10  Reconnaissance
9280803         44    None  ...  143.88.2.10  Reconnaissance
9280804         44    None  ...  143.88.2.10  Reconnaissance
9280805         44    None  ...  143.88.2.10  Reconnaissance

[9280806 rows x 23 columns]
resp_pkts                                      4
service                                     None
orig_ip_bytes                                176
local_resp                                 False
missed_bytes                                   0
proto                                        tcp
duration                                0.000845
conn_state                                   REJ
dest_ip_zeek                         143.88.2.12
orig_pkts                                      4
community_id      1:JCuToPyQMJ29Yj8yBq4IBtOuUxI=
resp_ip_bytes                                160
dest_port_zeek                               443
orig_bytes                                   0.0
local_orig                                 False
datetime              2022-02-11 09:52:26.618000
history                                       Sr
resp_bytes                                   0.0
uid                            CIk3hV1ZoelpIcgSg
src_port_zeek                              46411
ts                             1644573146.618264
src_ip_zeek                          143.88.7.10
label_tactic                           Discovery
Name: 0, dtype: object

Process finished with exit code 0

```

If you need more information about the datasets, feel free to contact me.

### Steps for building up the dataset:

1. Merge all the files into a single dataframe and remove the unusable input parameters:
    ```bash
    python3 ./threat-hunting-ia/threat_trekker.py --build-dataset -i uwf-dataset/ -o $(date +%FT%H-%M-%S)-uwf.parquet
    ```
2. Execute the script for performing the classification:
    ```bash
     python3 ./threat-hunting-ia/threat_trekker.py -i uwf-dataset/parsed/2023-07-09T16-04-31-uwf.parquet --sample 0.70
    ```

3. Execute the script for balancing the dataset and performing the classification:
    ```bash
     python3 ./threat-hunting-ia/threat_trekker.py -i uwf-dataset/parsed/2023-07-09T16-04-31-uwf.parquet --balance
    ```