# the c compiler
CC=g++

# directories
BLD_DIR=build
SRC_DIR=src
CODE_DIRS=$(shell find $(SRC_DIR) -maxdepth 16 -type d)

# compilation and linking flags
DEP_FLAGS= -MP -MD
#C_FLAGS  = -Wall
#C_FLAGS += -Wextra
#C_FLAGS += -Wpedantic
C_FLAGS += -g $(foreach D,$(CODE_DIRS),-I$(D)) $(DEP_FLAGS)
OPT_FLAGS= -O3 -c
DBG_FLAGS= -Og -c
ASM_FLAGS= -Og -S
LNK_FLAGS= -lm -fopenmp -lpthread -lginac -lcln

# files
BINARY_FILE=out
TA_FILE=tex_atlas
C_FILES=$(foreach D,$(CODE_DIRS),$(wildcard $(D)/*.cpp))
OBJECT_FILES=$(patsubst %.cpp,build/%.o,$(C_FILES))
DEPENDENCY_FILES=$(patsubst %.cpp,build/%.d,$(C_FILES))
COMPILED_ASM_FILES=$(patsubst %.cpp,build/%.s,$(C_FILES))

# compile
all: release

# compile heavily optimized for releases 
release: create_dirs opt_c_flags $(BINARY_FILE)

# compile for debugging purposes
debug: create_dirs dbg_c_flags $(BINARY_FILE)

# compile only into assembly language
asm: create_dirs imdasm_c_flags $(COMPILED_ASM_FILES)

# link all object files together to create a binary
$(BINARY_FILE):$(OBJECT_FILES)
	$(CC) -w -o $@ $^ $(LNK_FLAGS)

# compile a C-File into an Object-File
build/%.o:%.cpp
	$(CC) $(C_FLAGS) $(LNK_FLAGS) -o $@ $<

# compile a C-File into an Assembly File
build/%.s:%.cpp
	$(CC) $(C_FLAGS) -o $@ $<

# deletes all .o and .d files, the binary and the .tgz archive
clean:
	rm -f -r build $(BINARY_FILE) mechanics.tgz $(TA_FILE)

# create all needed directories
create_dirs:
	mkdir -p build $(foreach D,$(CODE_DIRS),$(BLD_DIR)/$(D))

tools:$(TOOLS_TA)
	$(CC) -o $(TA_FILE) $(TOOLS_TA) -lm

# gives info to all commands
help:
	$(info Available commands are: )
	$(info - 'make' will build the project in release mode.)
	$(info - 'make release' will build the project with optimization turned on.)
	$(info - 'make debug' will build the project with debug features turned on.)
	$(info - 'make asm' will only compile all source files into assembly files for inspection.)
	$(info - 'make clean' will delete all files needed for the building process in the /build directory.)
	$(info - 'make pack' will first clean and then pack all the source files into a tar-archive.)
	$(info - 'make count' will count up all written code lines using cloc.)
	$(info - 'make tools' will build all development tools written for the game.)
	$(info - 'make help' shows this message.)

# clean and pack all
pack: clean
	tar zcvf rational.tgz *

# populate the c flags with the extra opt flags
opt_c_flags:
	$(eval C_FLAGS += $(OPT_FLAGS))

# populate the c flags with the extra dbg flags
dbg_c_flags:
	$(eval C_FLAGS += $(DBG_FLAGS))

# populate the c flags with the extra asm flags
imdasm_c_flags:
	$(eval C_FLAGS += $(ASM_FLAGS))

# count lines of code
count:
	cloc --exclude-lang=JSON,D .

-include $(DEPENDENCY_FILES)
