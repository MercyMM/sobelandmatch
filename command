g++ -fPIC -DOPENCV `pkg-config --cflags --libs opencv`  -L/usr/local/cuda-6.5/lib -DGPU -Wall -Wfatal-errors -Ofast -DOPENCV -c
*.c -o *.o

cude:
/usr/local/cuda-6.5/bin/nvcc -w --gpu-architecture=compute_32 --gpu-code=compute_32 -DOPENCV `pkg-config --cflags --libs opencv` -I/usr/local/cuda-6.5/include/ --compiler-options "-Wall -Wfatal-errors -Ofast -fPIC" -c kernel.cu -o objs/kernel.o

kernel:
nvcc -shared -o kernel.so -L/usr/local/cuda-6.5/lib -lcuda -lcudart -lcublas -lcurand objs/kernel.o

g++ -c main.cpp -mfpu=neon `pkg-config --cflags --libs opencv` -o main.o

g++ -o main main.o ./kernel.so `pkg-config --cflags --libs opencv` -L/usr/local/cuda-6.5/lib -lcuda -lcudart -lcublas -lcurand


g++ -o main main.o objs/elas.o objs/descriptor.o objs/matrix.o objs/triangle.o ./kernel.so `pkg-config --cflags --libs opencv` -L/usr/local/cuda-6.5/lib -lcuda -lcudart -lcublas -lcurand
