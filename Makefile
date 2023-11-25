# Transition Time Analysis

TR_TIME_PLANS_DIR    := build/googlenet_transition_plans
TR_TIME_PROFILES_DIR := $(TR_TIME_PLANS_DIR)/profiles

# workaround as the python script generates all plans at once
TR_TIME_SENTINEL := $(TR_TIME_PLANS_DIR)/.sentinel

TR_TIME_PLANS := $(wildcard $(TR_TIME_PLANS_DIR)/*.plan)

$(TR_TIME_SENTINEL):
	python3 scripts/transition_time_analysis/build_transition_time_engines.py
	mkdir -p $(TR_TIME_PLANS_DIR) && touch $(TR_TIME_SENTINEL)

$(TR_TIME_PROFILES_DIR)/%.profile: $(TR_TIME_SENTINEL)
	mkdir -p $(TR_TIME_PROFILES_DIR)
	# The corresponding plan is found for each target profile here
	plan_file=$(patsubst $(TR_TIME_PROFILES_DIR)/%.profile,$(TR_TIME_PLANS_DIR)/%.plan,$@); \
	/usr/src/tensorrt/bin/trtexec --iterations=10000  \
	--exportProfile=$@ --avgRuns=1 --warmUp=5000 --duration=0 --loadEngine=$$plan_file

profiles: $(patsubst $(TR_TIME_PLANS_DIR)/%.plan,$(TR_TIME_PROFILES_DIR)/%.profile,$(TR_TIME_PLANS))

# EMC Analysis
EMC_PLANS_DIR := build/convolution_characterization_plans
EMC_TIMES_DIR := $(EMC_PLANS_DIR)/times

EMC_SENTINEL  := $(EMC_PLANS_DIR)/.sentinel
SRC_FILES := $(wildcard convolution_characterization_prototxts/*.prototxt)
EMC_PLANS := $(patsubst convolution_characterization_prototxts/%.prototxt,$(EMC_PLANS_DIR)/%.plan,$(SRC_FILES))
EMC_TIMES := $(patsubst $(EMC_PLANS_DIR)/%.plan,$(EMC_TIMES_DIR)/%.txt,$(EMC_PLANS))


$(EMC_SENTINEL):
	python3 scripts/emc_analysis/engine_build_convolution_characterization.py
	mkdir -p $(EMC_PLANS_DIR) && touch $(EMC_SENTINEL)

$(EMC_TIMES_DIR)/%.txt: $(EMC_SENTINEL)
	mkdir -p $(EMC_TIMES_DIR)
	sh scripts/emc_analysis/emc_single_run.sh $(EMC_PLANS_DIR)/$*.plan $@


output/emc_results.json: $(EMC_SENTINEL) $(EMC_TIMES)
	sudo python3 scripts/emc_analysis/emc_util_all.py

.PHONY: emc
emc: output/emc_results.json

all: emc

clean:
	rm -rf output/* build/*

.PHONY: all clean
