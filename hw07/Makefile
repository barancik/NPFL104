URL=http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/regression_data/

A1=artificial_2x_train.tsv
A2=artificial_2x_test.tsv

P1=pragueestateprices_train.tsv
P2=pragueestateprices_test.tsv

all: prague_train prague_test $(A1) $(A2)
	./regression.py

$(A1):
	wget -q $(URL)$(A1)

$(A2):
	wget -q $(URL)$(A2)

prague_train: $(P1)
	cat pragueestateprices_train.tsv | sed 's/[^\t]\+://g' | sed 's/Ano/1/g' | sed 's/Ne/0/g' | sed 's/Částečně/0.5/g' > prague_train

prague_test: $(P2)
	cat pragueestateprices_test.tsv | sed 's/[^\t]\+://g' | sed 's/Ano/1/g' | sed 's/Ne/0/g' | sed 's/Částečně/0.5/g' > prague_test

$(P1):
	wget -q $(URL)$(P1)

$(P2):
	wget -q $(URL)$(P2)

clean:
	rm -rf $(P2) $(P1) prague_test prague_train $(A2) $(A1)
