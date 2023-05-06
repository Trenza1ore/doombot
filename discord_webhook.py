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
        self.last_upload = []
        self.data = {}
        self.wait_time = wait_time
        self.epoch = -1
    
    def update_data_img(self, epoch_num):
        content = f"Epoch {epoch_num+1}"
        self.data = {
            "content" : content,
            "username" : "Doom Guy",
            "embeds" : [{
                "image" : {"url" : img},
                "type" : "rich"
                } for img in self.last_upload]
        }

    def send_img(self, epoch_num):
        self.last_upload = []
        file_names = [f"plots/{epoch_num}.png", f"plots/{epoch_num}a.png"]
        #if (epoch_num+1)%5 is 0:
        file_names.append("plots/train_quartiles.png")
        file_names.append("plots/train_kill_counts.png")
        for i in range(len(file_names)):
            with open(file_names[i], "rb") as file:
                url = "https://api.imgbb.com/1/upload"
                payload = {
                    "key" : IMGBB_KEY,
                    "image" : b64encode(file.read()),
                    "name" : f"test_{epoch_num:03d}"
                }
            try:
                result = requests.post(url, payload)
                self.last_upload.append(result.json()["data"]["display_url"])
            except:
                print(f"Unable to upload {file_names[i] %epoch_num}")
        try:
            self.update_data_img(epoch_num)
            result = requests.post(self.url, json = self.data)
        except:
            print(f"Unable to send images for epoch {epoch_num+1}")
    
    def send_stat(self, content):
        stat_data = {
            "content" : content,
            "username" : "Doom Guy"
        }
        try:
            result = requests.post(self.url, json = stat_data)
        except:
            print(f"Unable to send stats")