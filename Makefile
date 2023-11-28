# The first rule should be all

all: emc transition

# All targets will be protected and won't be deleted
.SECONDARY:

# Layer and Transition Time Analysis
# Layer analysis of DLA makes use of transitioning engines unlike GPU.

# Do not use 141 instead make use of -1 transition for single device run
TRANSITIONS := -1 10 25 39 53 67 81 95 110 124
PROTOTXT := prototxt_input_files/googlenet.prototxt


TR_TIME_PLANS_DIR    := build/googlenet_transition_plans
PLANS_GPU := $(foreach trans,$(TRANSITIONS),$(TR_TIME_PLANS_DIR)/googlenet_gpu_transition_at_$(trans).plan)
PLANS_DLA := $(foreach trans,$(TRANSITIONS),$(TR_TIME_PLANS_DIR)/googlenet_dla_transition_at_$(trans).plan)


TR_TIME_PROFILES_DIR := $(TR_TIME_PLANS_DIR)/profiles
TR_TIME_PROF_LOGS_DIR := $(TR_TIME_PLANS_DIR)/profile_logs
LOGS_GPU  := $(patsubst $(TR_TIME_PLANS_DIR)/%.plan, $(TR_TIME_PROF_LOGS_DIR)/%.log, $(PLANS_GPU))
LOGS_DLA  := $(patsubst $(TR_TIME_PLANS_DIR)/%.plan, $(TR_TIME_PROF_LOGS_DIR)/%.log, $(PLANS_DLA))

PROFILES_DLA := $(patsubst $(TR_TIME_PLANS_DIR)/%.plan, $(TR_TIME_PROFILES_DIR)/%.profile, $(PLANS_DLA))


$(TR_TIME_PLANS_DIR)/googlenet_gpu_transition_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --start gpu --output $@ --transition $* --verbose

$(TR_TIME_PLANS_DIR)/googlenet_dla_transition_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --start dla --output $@ --transition $* --verbose


$(TR_TIME_PROFILES_DIR)/%.profile $(TR_TIME_PROF_LOGS_DIR)/%.log: $(TR_TIME_PLANS_DIR)/%.plan
	mkdir -p $(TR_TIME_PROFILES_DIR) $(TR_TIME_PROF_LOGS_DIR)
	/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile \
	--exportProfile=$(TR_TIME_PROFILES_DIR)/$*.profile --avgRuns=1  \
	--warmUp=5000 --duration=0 --loadEngine=$< > $(TR_TIME_PROF_LOGS_DIR)/$*.log

# Layer Analysis Specifics

## GPU Layer Analysis
build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json: build/googlenet_transition_plans/profiles/googlenet_gpu_transition_at_-1.profile
	python3 scripts/layer_analysis/layer_gpu_util.py --profile build/googlenet_transition_plans/profiles/googlenet_gpu_transition_at_-1.profile

## DLA Layer Analysis
output/dla_compute_times.json: $(PROFILES_DLA)
	python3 scripts/layer_analysis/layer_dla_util.py --profiles_dir build/googlenet_transition_plans/profiles

## Collecting Layer Analysis Results
output/layer_results.json: build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json output/dla_compute_times.json
	python3 scripts/layer_analysis/layer_all_util.py --gpu_json build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json --dla_json output/dla_compute_times.json --output output/layer_results.json

.PHONY: layer
layer : output/layer_results.json


# Transition Analysis Specifics
output/transition_results.json: $(LOGS_GPU) $(LOGS_DLA)
	python3 scripts/transition_time_analysis/transition_util.py

.PHONY: transition
transition : output/transition_results.json


# EMC Analysis
PROTOTXT_DIR  := convolution_characterization_prototxts
EMC_PLANS_DIR := build/convolution_characterization_plans
EMC_TIMES_DIR := $(EMC_PLANS_DIR)/times

SRC_FILES := $(wildcard $(PROTOTXT_DIR)/*.prototxt)
EMC_PLANS := $(patsubst $(PROTOTXT_DIR)/%.prototxt,$(EMC_PLANS_DIR)/%.plan,$(SRC_FILES))
EMC_TIMES := $(patsubst $(EMC_PLANS_DIR)/%.plan,$(EMC_TIMES_DIR)/%.txt,$(EMC_PLANS))


$(EMC_PLANS_DIR)/%.plan: $(PROTOTXT_DIR)/%.prototxt
	mkdir -p $(EMC_PLANS_DIR)
	python3 src/build_engine.py --prototxt $< --output $@ --start gpu

$(EMC_TIMES_DIR)/%.txt: $(EMC_PLANS_DIR)/%.plan
	mkdir -p $(EMC_TIMES_DIR)
	sh scripts/emc_analysis/emc_single_run.sh $< $@


output/emc_results.json: $(EMC_TIMES)
	mkdir -p output
	sudo python3 scripts/emc_analysis/emc_util_all.py

.PHONY: emc
emc: output/emc_results.json


clean:
	rm -rf output/* build/*

.PHONY: all clean
