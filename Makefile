

#
#
# Author: Teunis van Beelen
#
# email: teuniz@gmail.com
#
#

CC = gcc
CFLAGS = -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -o2 

# objects = rs232.o

all: test_rx test_tx

build: main.o camerasubtractor.o modelSVM.o findLinkPixel.o rs232.o
	g++ main.o camerasubtractor.o modelSVM.o findlinkpixel.o rs232.o -o output `pkg-config --cflags --libs opencv`


rs232.o : rs232.h rs232.c
	$(CC) $(CFLAGS) -c rs232.c -o rs232.o


main.o: main.cpp
	g++ -c main.cpp

camerasubtractor.o: camerasubtractor.h camerasubtractor.cpp
	g++ -c camerasubtractor.cpp 

modelSVM.o: modelSVM.h modelSVM.cpp
	g++ -c modelSVM.cpp

findLinkPixel.o: findlinkpixel.h findlinkpixel.cpp
	g++ -c findlinkpixel.cpp

clean:
	rm -rf *.o output

run:	
	./output -pd=data_train/Person/ -nd=data_train/NonePerson/ -vid=video_1.avi -kn=2 -c=10 -g=0.1 -ft=false -fn=model.dat -b_w=10 -b_h=20 -bs_w=5 -bs_h=10 -c_w=10 -c_h=10 -n_bin=9
