.PHONY : install install-dev install-all test check build docs clean push_release

CHECK_PATHS = \
	spateo/_native \
	spateo/io \
	spateo/preprocessing \
	spateo/alignment/utils.py \
	spateo/plotting/static/position.py \
	spateo/svg/get_svg.py \
	spateo/svg/get_svg_between_slice.py \
	spateo/svg/utils.py \
	spateo/tdr/interpolations/interpolation_sparseVFC.py \
	spateo/tdr/models/models_backbone/backbone.py \
	spateo/tdr/models/models_backbone/backbone_methods.py \
	spateo/tdr/models/models_migration/arrow_model.py \
	spateo/tdr/models/models_migration/morphofield_model.py \
	spateo/tdr/models/models_migration/morphopath_model.py \
	spateo/tdr/morphometrics/morphofield/sparsevfc.py \
	spateo/tdr/morphometrics/morphofield/trajectory.py \
	spateo/tdr/morphometrics/morphofield_dg/differential_geometry.py \
	spateo/tools/cluster/cluster_spagcn.py \
	spateo/tools/cluster/spagcn_utils.py \
	spateo/tools/cluster/utils.py \
	spateo/tools/glm.py \
	spateo/tools/spatially_variable_gene_ot.py \
	tests/io \
	tests/preprocessing \
	tests/test_native_runtime.py \
	skills/setup-spateo-environment/scripts/verify_environment.py \
	spateo/get_version.py

install:
	pip install .
	# There is a problem with just pip installing hdbscan...
	# pip uninstall -y hdbscan
	# pip install --no-build-isolation --no-binary :all: hdbscan>=0.8.26

install-dev:
	pip install -r dev-requirements.txt

install-tdr:
	pip install -r 3d-requirements.txt

install-docs:
	pip install -r docs/requirements.txt

install-all: install-dev install-docs install

test:
	pytest --verbose --cov=spateo tests

check:
	python -m compileall -q spateo
	isort --profile black --check $(CHECK_PATHS)
	black --check $(CHECK_PATHS)
	@echo OK

build:
	python setup.py sdist

docs:
	sphinx-build -a docs docs/_build

clean:
	rm -rf build
	rm -rf dist
	rm -rf spateo.egg-info
	rm -rf docs/_build
	rm -rf docs/autoapi
	rm -rf .coverage

bump_patch:
	bumpversion patch

bump_minor:
	bumpversion minor

bump_major:
	bumpversion major

push_release:
	git push && git push --tags
