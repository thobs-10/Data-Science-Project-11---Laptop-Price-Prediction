import json
import uuid
from datetime import datetime
from time import sleep

import pyarrow.parquet as pq
import requests

table = pq.read_table("dataset\\feature_engineered_data\\X_test_df.parquet.gzip")
data = table.to_pylist()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

# read the target csv file
with open("predicted_target.csv", 'w') as f_target:
    # go through each row
    for row in data:
        # create a unique id
        row['id'] = str(uuid.uuid4())
        resp = requests.post("http://127.0.0.1:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(row, cls=DateTimeEncoder)).json()
         # print the price value from the response of the request from the endpoint
        print(f"prediction: {resp['price']}")
        # sleep for 1 second and run again for another row
        sleep(1)
