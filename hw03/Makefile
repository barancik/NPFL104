DATA_FILE=artificial_objects.tgz
URL=http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/$(DATA_FILE)

all: $(DATA_FILE)
	ls -lt   

$(DATA_FILE):
	wget $(URL)
	tar xzf $(DATA_FILE)

