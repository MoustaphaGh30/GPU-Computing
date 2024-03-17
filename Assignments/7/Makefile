
NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.obj kernel.obj
EXE         = scan


default: $(EXE)

%.obj: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

