
ll:
	gcc -shared -fPIC nn.c -o nn.so -lm

debug:
	gcc -shared -fPIC nn.c -o nn.so -lm -ggdb

