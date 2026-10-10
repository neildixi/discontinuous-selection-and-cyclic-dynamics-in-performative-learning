PYTHON ?= python
WORKERS ?= 2

.PHONY: figures notebooks quick reproduce-paper reproduce-full test manifest verify-cached

figures:
	$(PYTHON) -m dynafront.cli figures

notebooks:
	$(PYTHON) -m dynafront.cli notebooks

quick:
	$(PYTHON) -m dynafront.cli reproduce --mode quick --workers $(WORKERS)

reproduce-paper:
	$(PYTHON) -m dynafront.cli reproduce --mode paper --workers $(WORKERS)

reproduce-full:
	$(PYTHON) -m dynafront.cli reproduce --mode full --workers $(WORKERS) --skip-if-over-minutes 45

test:
	$(PYTHON) -m pytest

manifest:
	$(PYTHON) -m dynafront.cli manifest

verify-cached:
	$(PYTHON) -m dynafront.cli verify-cached
