D1=credit.train.txt
D2=credit.test.txt

A3=AESOP.v2.encrypted
A4=aesop.data

C=clouds_dataset.zip
C1=clouds_dataset/clouds1k_test.csv 
C2=clouds_dataset/clouds1k_train.csv

AESOP=https://atrey.karlin.mff.cuni.cz/~hajicja/AESOP_dataset/$(A3)
URL1=http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/
URL3=http://archive.ics.uci.edu/ml/machine-learning-databases/adult/

all: $(D1) $(D2) $(C1) $(A4)
	./plot_curves.py

$(D1):
	wget $(URL1)$(D1)

$(D2):
	wget $(URL1)$(D2)

$(C1): $(C2)
	

$(C2):
	wget $(URL1)$(C)
	unzip -q $(C)

$(A4):
	wget $(AESOP)
	cat $(A3) | crypt | sed 's/,$$//' > $(A4)

clean:
	rm -rf [aA]*
	rm -rf clouds_dataset*
	rm -rf credit.*
