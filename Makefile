.PHONY: all clean

all:
	$(MAKE) -C c_kernel

clean:
	$(MAKE) -C c_kernel clean
