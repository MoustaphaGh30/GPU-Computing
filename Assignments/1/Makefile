
NVCC    = nvcc
OBJ     = main.obj kernel.obj
EXE     = vecmax


default: $(EXE)

%.obj: %.cu
	$(NVCC) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

