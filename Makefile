# The first rule should be all

all: emc transition layer

# All targets will be protected and won't be deleted
.SECONDARY:

# Layer and Transition Time Analysis
# Layer analysis of DLA makes use of transitioning engines unlike GPU.
# But Transition Analysis does not use transitioning engines instead it
# uses engines with layers marked as output at transition points.

# Do not use 141 instead make use of -1 transition for single device run
TRANSITIONS := -1 10 24 38 53 67 81 95 110 124
PROTOTXT := prototxt_input_files/googlenet.prototxt

MARK_PLANS_DIR    := build/googlenet_mark_plans
MARK_PLANS_GPU := $(foreach trans,$(TRANSITIONS),$(MARK_PLANS_DIR)/googlenet_gpu_mark_at_$(trans).plan)
MARK_PLANS_DLA := $(foreach trans,$(TRANSITIONS),$(MARK_PLANS_DIR)/googlenet_dla_mark_at_$(trans).plan)

$(MARK_PLANS_DIR)/googlenet_gpu_mark_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --start gpu --output $@ --mark $* --verbose

$(MARK_PLANS_DIR)/googlenet_dla_mark_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --start dla --output $@ --mark $* --verbose

TR_TIME_PLANS_DIR    := build/googlenet_transition_plans
TR_PLANS_GPU := $(foreach trans,$(TRANSITIONS),$(TR_TIME_PLANS_DIR)/googlenet_gpu_transition_at_$(trans).plan)
TR_PLANS_DLA := $(foreach trans,$(TRANSITIONS),$(TR_TIME_PLANS_DIR)/googlenet_dla_transition_at_$(trans).plan)


$(TR_TIME_PLANS_DIR)/googlenet_gpu_transition_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --start gpu --output $@ --transition $* --verbose

$(TR_TIME_PLANS_DIR)/googlenet_dla_transition_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --start dla --output $@ --transition $* --verbose


MARK_PROFILES_DIR := $(MARK_PLANS_DIR)/profiles
MARK_PROF_LOGS_DIR := $(MARK_PLANS_DIR)/profile_logs
MARK_LOGS_GPU  := $(patsubst $(MARK_PLANS_DIR)/%.plan, $(MARK_PROF_LOGS_DIR)/%.log, $(MARK_PLANS_GPU))
MARK_LOGS_DLA  := $(patsubst $(MARK_PLANS_DIR)/%.plan, $(MARK_PROF_LOGS_DIR)/%.log, $(MARK_PLANS_DLA))

MARK_PROFILES_DLA := $(patsubst $(MARK_PLANS_DIR)/%.plan, $(MARK_PROFILES_DIR)/%.profile, $(MARK_PLANS_DLA))

$(MARK_PROFILES_DIR)/%.profile $(MARK_PROF_LOGS_DIR)/%.log: $(MARK_PLANS_DIR)/%.plan
	mkdir -p $(MARK_PROFILES_DIR) $(MARK_PROF_LOGS_DIR)
	/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile \
	--exportProfile=$(MARK_PROFILES_DIR)/$*.profile --avgRuns=1  \
	--warmUp=5000 --duration=0 --loadEngine=$< > $(MARK_PROF_LOGS_DIR)/$*.log


TR_TIME_PROFILES_DIR := $(TR_TIME_PLANS_DIR)/profiles
TR_TIME_PROF_LOGS_DIR := $(TR_TIME_PLANS_DIR)/profile_logs
TR_LOGS_GPU  := $(patsubst $(TR_TIME_PLANS_DIR)/%.plan, $(TR_TIME_PROF_LOGS_DIR)/%.log, $(TR_PLANS_GPU))
TR_LOGS_DLA  := $(patsubst $(TR_TIME_PLANS_DIR)/%.plan, $(TR_TIME_PROF_LOGS_DIR)/%.log, $(TR_PLANS_DLA))

TR_PROFILES_DLA := $(patsubst $(TR_TIME_PLANS_DIR)/%.plan, $(TR_TIME_PROFILES_DIR)/%.profile, $(TR_PLANS_DLA))
TR_PROFILES_GPU := $(patsubst $(TR_TIME_PLANS_DIR)/%.plan, $(TR_TIME_PROFILES_DIR)/%.profile, $(TR_PLANS_GPU))

$(TR_TIME_PROFILES_DIR)/%.profile $(TR_TIME_PROF_LOGS_DIR)/%.log: $(TR_TIME_PLANS_DIR)/%.plan
	mkdir -p $(TR_TIME_PROFILES_DIR) $(TR_TIME_PROF_LOGS_DIR)
	/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile \
	--exportProfile=$(TR_TIME_PROFILES_DIR)/$*.profile --avgRuns=1  \
	--warmUp=5000 --duration=0 --loadEngine=$< > $(TR_TIME_PROF_LOGS_DIR)/$*.log

# Layer Analysis Specifics
LAYER_DIR := $(TR_TIME_PLANS_DIR)/layer_times
ONLY_GPU := $(TR_TIME_PROFILES_DIR)/googlenet_gpu_transition_at_-1.profile

## GPU Layer Analysis
$(LAYER_DIR)/googlenet_gpu_transition_at_-1_filtered.json: $(ONLY_GPU)
	python3 scripts/layer_analysis/layer_gpu_util.py --profile $(ONLY_GPU)

## DLA Layer Analysis
# DLA analysis looks at transitions that start from gpu and continue with dla
output/dla_compute_times.json: $(TR_PROFILES_GPU) $(ONLY_GPU)
	python3 scripts/layer_analysis/layer_dla_util.py --profiles_dir build/googlenet_transition_plans/profiles

## Collecting Layer Analysis Results
output/layer_results.json: build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json output/dla_compute_times.json
	python3 scripts/layer_analysis/layer_all_util.py --gpu_json build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json --dla_json output/dla_compute_times.json --output output/layer_results.json

.PHONY: layer
layer : output/layer_results.json


# Transition Analysis Specifics
output/transition_results.json: $(MARK_LOGS_GPU) $(MARK_LOGS_DLA)
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
	rm -rf output/* build/* starter_guide_logs/*

.PHONY: all clean
