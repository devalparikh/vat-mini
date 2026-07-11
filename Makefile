PYTHON ?= python3.12
VENV := .venv
PIP := $(VENV)/bin/pip
VAT := $(VENV)/bin/vat-mini

.PHONY: setup setup-tracking setup-robotics inspect test smoke data postdata pretrain posttrain robomimic-can evaluate learn learn-build clean-runs

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e '.[dev]'

setup-tracking: setup
	$(PIP) install -e '.[dev,tracking]'

setup-robotics: setup
	$(PIP) install -e '.[dev,robotics]'

inspect:
	$(VAT) inspect

test:
	$(VENV)/bin/pytest

smoke:
	$(VAT) smoke --config configs/smoke.yaml

data:
	$(VAT) generate-data --config configs/pretrain.yaml

postdata:
	$(VAT) generate-data --config configs/posttrain.yaml

pretrain: data
	$(VAT) train --config configs/pretrain.yaml

posttrain: postdata
	$(VAT) train --config configs/posttrain.yaml

robomimic-can:
	$(VAT) train --config configs/robomimic-can.yaml

evaluate:
	$(VAT) evaluate --config configs/posttrain.yaml

learn:
	npm --prefix learning-site install
	npm --prefix learning-site run dev

learn-build:
	npm --prefix learning-site install
	npm --prefix learning-site run build

clean-runs:
	rm -rf runs data/generated data/gridworld.npz
