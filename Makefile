SENTINEL=build/convolution_characterization_plans/.sentinel

$(SENTINEL):
	python3 scripts/emc_analysis/engine_build_convolution_characterization.py
	touch $(SENTINEL)

output/emc_results.yaml: $(SENTINEL)
	python3 scripts/emc_analysis/emc_util_all.py

all: output/emc_results.yaml

clean:
	rm -r output/* build/*

.PHONY: all clean
