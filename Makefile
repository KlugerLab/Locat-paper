.PHONY: publish-tutorial

# Execute the tutorial notebook and run regression tests
run-and-test-tutorial:
	pytest tests/test_pbmc_tutorial_repro.py -v --run-notebook

# Rerun the tutorial on jupyterlab with light mode to get proper progress-bar output before running this.
# Checks the results and copies to the Locat project
publish-tutorial:
	pytest tests/test_pbmc_tutorial_repro.py -v
	@if [ -d ../Locat ]; then \
		cp notebooks/locat_tutorial_pbmc3k/locat_tutorial_pbmc3k.ipynb \
		   ../Locat/docs/source/tutorials/pbmc3k.ipynb; \
		echo "Done. Commit ../Locat/docs/source/tutorials/pbmc3k.ipynb before releasing."; \
	else \
		echo "Skipping copy: ../Locat not found."; \
	fi
