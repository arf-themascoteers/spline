import pandas as pd
import numpy as np

cols = ["id","oc","phh","phc","ec","caco3","p","n","k","elevation","stones","lc1","lu1",
        "443","490","560","665","705","740","783","842","865","940","1375","1610","2190"]
df = pd.read_csv("data/dataset_r.csv")
df = df[cols]
df.rename(columns={
    "443": "b1",
    "490": "blue",
    "560": "green",
    "665": "red",
    "705": "vnir1",
    "740": "vnir2",
    "783": "vnir3",
    "842": "vnir4",
    "865": "vnir5",
    "940": "swir1",
    "1375": "swir2",
    "1610": "swir3",
    "2190": "swir4"
    }, inplace=True)
df.to_csv("data/dataset_s2.csv", index=False)