from base64 import b64encode
import requests
import traceback
import os
import time
import matplotlib.pyplot as plt

WEBHOOK_URL = "***REMOVED***"
IMGBB_KEY = "***REMOVED***"

class discord_bot:
    def __init__(self, url: str = WEBHOOK_URL, extra: str=''):
        current_time = time.strftime("%y-%m-%d-%H-%M-%S")
        self.url = url
        self.last_upload = []
        self.data = {}
        self.epoch = -1
        self.path = "history/" + current_time
        os.mkdir(self.path)
        self.send_string("**New Run: ** " + current_time + " " + extra)
    
    def update_data_img(self, epoch_num: int):
        content = f"Epoch {epoch_num+1}"
        self.data = {
            "content" : content,
            "username" : "Doom Guy",
            "embeds" : [{
                "image" : {"url" : img},
                "type" : "rich"
                } for img in self.last_upload]
        }

    def send_img(self, epoch_num: int):
        self.last_upload = []
        
        # file_names = [f"{self.path}/{epoch_num}.png", 
        #               f"{self.path}/{epoch_num}a.png", 
        #               f"{self.path}/train_quartiles.png", 
        #               f"{self.path}/train_kill_counts.png"]
        
        # plt.clf()
        # for i in range(4):
        #     plt.subplot(2, 2, i+1)
        #     plt.imshow(plt.imread(file_names[i]))
        # plt.savefig("plots/current.png", dpi=100)
        
        with open(f"{self.path}/current.png", "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {
                "key" : IMGBB_KEY,
                "image" : b64encode(file.read()),
                "name" : f"test{self.path}"
            }
            
        try:
            result = requests.post(url, payload)
            self.last_upload.append(result.json()["data"]["display_url"])
        except:
            print(f"Unable to upload images for epoch {epoch_num+1}")
            
        try:
            self.update_data_img(epoch_num)
            requests.post(self.url, json = self.data)
        except:
            print(f"Unable to send images for epoch {epoch_num+1}")
    
    def send_string(self, content: str):
        stat_data = {
            "content" : content,
            "username" : "Doom Guy"
        }
        try:
            requests.post(self.url, json = stat_data)
        except:
            print(f"Unable to send stats")
    
    def send_error(self, e: Exception):
        err_msg = ''.join(traceback.format_exception(e))
        self.send_string(err_msg)
        print(err_msg)