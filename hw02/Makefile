DATA_FILE=subject101.dat.gz
URL=http://ufallab.ms.mff.cuni.cz/~bojar/NPFL104/$(DATA_FILE)

all: $(DATA_FILE)
	./plot.py

$(DATA_FILE):
	wget $(URL)

