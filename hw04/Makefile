D1=credit.train.txt
D2=credit.test.txt

A1=adult.data
A2=adult.test

c=clouds_dataset
cloud_data = $(c)/clouds100_test.csv $(c)/clouds1k_test.csv $(c)/clouds100_train.csv $(c)/clouds1k_train.csv

AESOP=https://atrey.karlin.mff.cuni.cz/~hajicja/AESOP_dataset/AESOP.v2.encrypted

URL1=http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/
URL2=http://github.com/rdk/mlx/blob/master/datasets/$(c).zip?raw=true
URL3=http://archive.ics.uci.edu/ml/machine-learning-databases/adult/

#URL=http://ufallab.ms.mff.cuni.cz/~bojar/NPFL104/$(DATA_FILE)

all: $(D1) $(D2) $(cloud_data)
	ls -lt

$(D1):
	wget $(URL1)$(D1)

$(D2):
	wget $(URL1)$(D2)

$(cloud_data): $(c).zip
	unzip -q $(c).zip

$(c).zip:
	echo "Fake"
	wget $(URL2)
