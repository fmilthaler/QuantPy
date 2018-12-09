include common.mk

PYDIR=qpy
DATADIR=data
EXAMPLEDIR=example
CLEANDIRS = $(PYDIR:%=clean-%) $(EXAMPLEDIR:%=clean-%)

SEARCH=

.PHONY: cleandirs $(CLEANDIRS)
.PHONY: data
.PHONY: clean

all: clean

data:
	@$(MAKE) -C data convert

clean: $(CLEANDIRS)
$(CLEANDIRS):
	@echo "cleaning directory $(@:clean-%=%):"
	@$(MAKE) -C $(@:clean-%=%) clean

search:
	@echo "searching all python files for $(SEARCH):"
	@find . -name "*.py" | xargs grep -i --color=auto $(SEARCH)

