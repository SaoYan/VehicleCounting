INCLUDE = $(shell pkg-config opencv-3 --cflags)
LIBS = $(shell pkg-config opencv-3 --libs)

all: DVL

DVL: main.o
	g++ -o DVL main.o $(LIBS) -lpython2.7

main.o: main.cpp matplotlibcpp.h
	g++ -c main.cpp $(INCLUDE) -I/usr/include/python2.7

clean:
	rm -f *.o DVL
