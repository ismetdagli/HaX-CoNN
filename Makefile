# Transition Time Analysis

TR_TIME_PLANS_DIR    := build/googlenet_transition_plans
TR_TIME_PROFILES_DIR := $(TR_TIME_PLANS_DIR)/profiles
TR_TIME_PROF_LOGS_DIR := $(TR_TIME_PLANS_DIR)/profile_logs

# workaround as the python script generates all plans at once
TR_TIME_SENTINEL := $(TR_TIME_PLANS_DIR)/.sentinel


$(TR_TIME_SENTINEL):
	python3 scripts/transition_time_analysis/build_transition_time_engines.py
	mkdir -p $(TR_TIME_PLANS_DIR) && touch $(TR_TIME_SENTINEL)

$(TR_TIME_PROFILES_DIR)/%.profile: $(TR_TIME_SENTINEL)
	mkdir -p $(TR_TIME_PROFILES_DIR) $(TR_TIME_PROF_LOGS_DIR)
	# The corresponding plan is found for each target profile here
	plan_file=$(patsubst $(TR_TIME_PROFILES_DIR)/%.profile,$(TR_TIME_PLANS_DIR)/%.plan,$@); \
	log_file=$(patsubst $(TR_TIME_PROFILES_DIR)/%.profile,$(TR_TIME_PROF_LOGS_DIR)/%.log,$@); \
	/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile \
	--exportProfile=$@ --avgRuns=1 --warmUp=5000 --duration=0 --loadEngine=$$plan_file > $$log_file

.PHONY: profiles
profiles: $(TR_TIME_SENTINEL)
	$(eval TR_TIME_PLANS := $(wildcard $(TR_TIME_PLANS_DIR)/*.plan))
	$(MAKE) $(patsubst $(TR_TIME_PLANS_DIR)/%.plan,$(TR_TIME_PROFILES_DIR)/%.profile,$(TR_TIME_PLANS))


# EMC Analysis
PROTOTXT_DIR  := convolution_characterization_prototxts
EMC_PLANS_DIR := build/convolution_characterization_plans
EMC_TIMES_DIR := $(EMC_PLANS_DIR)/times

SRC_FILES := $(wildcard $(PROTOTXT_DIR)/*.prototxt)
EMC_PLANS := $(patsubst $(PROTOTXT_DIR)/%.prototxt,$(EMC_PLANS_DIR)/%.plan,$(SRC_FILES))
EMC_TIMES := $(patsubst $(EMC_PLANS_DIR)/%.plan,$(EMC_TIMES_DIR)/%.txt,$(EMC_PLANS))


$(EMC_PLANS_DIR)/%.plan: $(PROTOTXT_DIR)/%.prototxt
	mkdir -p $(EMC_PLANS_DIR)
	python3 src/build_engine.py --prototxt $< --output $@ --starts_gpu True

$(EMC_TIMES_DIR)/%.txt: $(EMC_PLANS_DIR)/%.plan
	mkdir -p $(EMC_TIMES_DIR)
	sh scripts/emc_analysis/emc_single_run.sh $< $@


output/emc_results.json: $(EMC_TIMES)
	sudo python3 scripts/emc_analysis/emc_util_all.py

.PHONY: emc
emc: output/emc_results.json

all: emc profiles

clean:
	rm -rf output/* build/*

.PHONY: all clean
