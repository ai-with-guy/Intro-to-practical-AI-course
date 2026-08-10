SOURCE_DIR := ../numpy-2.0-solution
COPIED_FILES := .venv numpy_hello*.so requirements.txt build.py numpy_hello.c

.PHONY: all clean

all:
	sh "$(SOURCE_DIR)/solve.sh"
	rm -rf .venv
	rm -f numpy_hello*.so
	cp -R "$(SOURCE_DIR)/.venv" .
	cp "$(SOURCE_DIR)"/numpy_hello*.so .
	cp "$(SOURCE_DIR)/requirements.txt" .
	cp "$(SOURCE_DIR)/build.py" .
	cp "$(SOURCE_DIR)/numpy_hello.c" .
	@echo "Copied the locally built program into docker-exercise."

clean:
	rm -rf $(COPIED_FILES)
