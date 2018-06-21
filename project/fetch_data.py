import urllib3 as url
response = url.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')
html = response.read()
