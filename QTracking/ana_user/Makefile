CC=g++
CFLAGS=-c -g -Wall -w `root-config --cflags`
LDFLAGS=`root-config --glibs`
GFORT=
SOURCES=ana_user.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=ana_user
 all: $(SOURCES) $(EXECUTABLE)
 $(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) $(GFORT)
 .cc.o:
	$(CC) $(CFLAGS) $< -o $@ $(CFLAGS)
 clean:
	rm ./*.o ./ana_user


