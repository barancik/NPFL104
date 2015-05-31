DATA_FILE=artificial_objects.tgz
A1=adult.data
A2=adult.test

URL=http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/

all: $(DATA_FILE) $(A1) $(A2)
	./perceptron.py   

$(DATA_FILE):
	wget $(URL)$(DATA_FILE)
	tar xzf $(DATA_FILE)
$(A1):
	wget $(URL)$(A1)

$(A2):
	wget $(URL)$(A2)

