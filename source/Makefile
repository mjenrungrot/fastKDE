OBJS	= experiments.o kdtree.o kernel_help.o
SOURCE	= experiments.cpp kdtree.cpp kernel_help.cpp
HEADER	= kdtree.h
OUT	= experiments
CC	 = g++
FLAGS	 = -g -c 
LFLAGS	 = 

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

experiments.o: experiments.cpp kdtree.o kernel_help.o
	$(CC) $(FLAGS) experiments.cpp 

kdtree.o: kdtree.cpp
	$(CC) $(FLAGS) kdtree.cpp 

kernel_help.o: kernel_help.cpp
	$(CC) $(FLAGS) kernel_help.cpp 


clean:
	rm -f $(OBJS) $(OUT)