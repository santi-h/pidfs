TARGET = a.exe
OBJECTS = 	main.obj \
            State.obj \
            StatePresenter.obj \
            start_states.obj \
            misc.obj \
            Log.obj \
            kernel.obj \
            ida.obj \
            bfs.obj \
            compute_cutoff.obj


.SUFFIXES : .cu .obj 
IGNOREW = -wd4514,-wd4505,-wd4820,-wd4365,-wd4986,-wd4710,-wd4668
# ,-wd4512,-wd4324,-wd4640,-wd4571,-wd4347,-wd4987,-wd4515,-wd4018,-wd4626,-wd4127,-wd4191,-wd4100
OPTIONS = -Xcompiler=-Wall,$(IGNOREW) -g -arch=sm_20 -lcudalog -lineinfo

# LINK OBJECTS
$(TARGET): $(OBJECTS)
    @echo linking $@...
	@nvcc $(OPTIONS) -Xcompiler=-wd4100 -o=$@ $**

# CREATE OBJECTS
.cu.obj: 
    @echo compiling $@...
	@nvcc $(OPTIONS) -dc -o=$@ "$*.cu"

# RUN
run: $(TARGET)
	@cls
	@.\$(TARGET)

# CLEAN
clean: 
	del .\$(TARGET)
	del .\*.obj

# REBUILD
rebuild: clean $(TARGET)
