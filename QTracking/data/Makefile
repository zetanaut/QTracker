CC=g++
CFLAGS=-c -g -Wall -w `root-config --cflags`
LDFLAGS=`root-config --glibs` -lMLP -lMinuit -lTreePlayer -lTMVA -lTMVAGui -lXMLIO  -lMLP -lm
GFORT=
SOURCES=Datamerge.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=Datamerge
 all: $(SOURCES) $(EXECUTABLE)
 $(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) $(GFORT)
 .cc.o:
	$(CC) $(CFLAGS) $< -o $@ $(CFLAGS)
 clean:
	rm ./*.o ./Datamerge


