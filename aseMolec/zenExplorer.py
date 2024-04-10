import os
import re
import time
import requests
import zipfile
from tqdm import tqdm

class ze:

    def __init__(self, ACCESS_TOKEN, recIDs):
        self.ACCESS_TOKEN = ACCESS_TOKEN
        self.recIDs = recIDs
        self.urls = dict()
        for recID in self.recIDs:
            r = requests.get('https://zenodo.org/api/deposit/depositions/'+str(recID), params={'access_token': self.ACCESS_TOKEN})
            self.urls.update({db['filename']:db['links']['download'] for db in r.json()['files']})
        self.urls = dict(sorted(self.urls.items(), key=lambda x: x[1]))
    
    def get_zip(self, fname):
        fbase = os.path.splitext(fname)[0]
        url = self.urls[fname]
        recID = int(re.search(r"/records/(\d+)/", url).group(1))
        os.makedirs('.cache', exist_ok=True)
        if not os.path.exists('.cache/'+str(recID)+'/'+fbase):
            os.makedirs('.cache/'+str(recID), exist_ok=True)
            file_response = requests.get(self.urls[fname], params={'access_token': self.ACCESS_TOKEN})
            with open('.cache/'+str(recID)+'/'+fname, 'wb') as f:
                f.write(file_response.content)
            with zipfile.ZipFile('.cache/'+str(recID)+'/'+fname, 'r') as zip_ref:
                zip_ref.extractall('.cache/'+str(recID)+'/'+fbase)
            os.remove('.cache/'+str(recID)+'/'+fname)
        else:
            time.sleep(0.1)

    def cache_all_data(self):
        for url_key in tqdm(self.urls):
            self.get_zip(url_key)
