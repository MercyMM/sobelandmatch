
TARGET = elas

#SRCS = $(wildcard *.cpp)

SRCS = main.cpp\
	elas.cpp\
	descriptor.cpp\
	matrix.cpp\
	triangle.cpp\
	vibe.cpp

OBJS = $(SRCS:.cpp=.o)

CC = g++

#CXXFLAGS = -mfpu=neon -fpermissive -w -O3
CXXFLAGS = -mfpu=neon -fpermissive -w -g
LDFLAGS = `pkg-config --cflags --libs opencv` 

all : bin/$(TARGET)
	@echo $(SRCS)

bin/$(TARGET) : $(addprefix objs/, $(OBJS))
	$(CC) -o $@ $(CXXFLAGS) $^ ./kernel.so $(LDFLAGS) -L/usr/local/cuda-6.5/lib 

#objs/%.o : %.cpp
#	$(CC) $(CXXFLAGS) -o $@ -c $<

objs/main.o:main.cpp
	$(CC) $(CXXFLAGS) -o $@ -c $<

objs/elas.o:elas.cpp
	$(CC) $(CXXFLAGS) -o $@ -c $<

objs/descriptor.o:descriptor.cpp
	$(CC) $(CXXFLAGS) -o $@ -c $<

objs/matrix.o:matrix.cpp
	$(CC) $(CXXFLAGS) -o $@ -c $<

objs/triangle.o:triangle.cpp
	$(CC) $(CXXFLAGS) -o $@ -c $<

objs/vibe.o:vibe.cpp
	$(CC) $(CXXFLAGS) -o $@ -c $<

clean:
	rm objs/main.o objs/triangle.o objs/elas.o objs/matrix.o objs/descriptor.o
