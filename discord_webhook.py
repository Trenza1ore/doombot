from json import loads as jsonload, dumps
from base64 import b64encode
import requests
from threading import Thread
from time import sleep


WEBHOOK_URL = "***REMOVED***"
IMGBB_KEY = "***REMOVED***"

class discord_bot:
    def __init__(self, wait_time: int = 60, url: str = WEBHOOK_URL):
        self.url = url
        self.last_upload = ""
        self.data = {}
        self.wait_time = wait_time
        self.epoch = -1
    
    def update_data_img(self, epoch_num, i):
        content = [f"Epoch {epoch_num}", "Overall", "Epoch Quartiles"]
        self.data = {
            "content" : content[i],
            "username" : "Doom Guy",
            "embeds" : [{
                "image" : {"url" : self.last_upload},
                "type" : "rich"
                }]
        }

    def send_img(self, epoch_num):
        file_names = [f"plots/{epoch_num}.png", f"plots/{epoch_num}a.png"]
        if (epoch_num+1)%10 is 0:
            file_names.append("plots/epoch_quartiles.png")
        for i in range(len(file_names)):
            with open(file_names[i], "rb") as file:
                url = "https://api.imgbb.com/1/upload"
                payload = {
                    "key" : IMGBB_KEY,
                    "image" : b64encode(file.read()),
                    "name" : f"test_{epoch_num:03d}"
                }
            result = requests.post(url, payload)
            self.last_upload = result.json()["data"]["display_url"]
            self.update_data_img(epoch_num, i)
            result = requests.post(self.url, json = self.data)
            try:
                result.raise_for_status()
            except requests.exceptions.HTTPError as err:
                print(err)
            else:
                print("Payload delivered successfully, code {}.".format(result.status_code))