import os

os.system('mkdir -p data/raw')
os.system('wget -O data/raw/covid_train.csv https://www.dropbox.com/s/lmy1riadzoy0ahw/covid.train.csv?dl=0')
os.system('wget -O data/raw/covid_test.csv https://www.dropbox.com/s/zalbw42lu4nmhr2/covid.test.csv?dl=0')
