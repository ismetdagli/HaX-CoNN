# Transition Time Analysis

TRANSITIONS := -1 0 10 24 38 53 67 81 95 109 124 141
PROTOTXT := prototxt_input_files/googlenet.prototxt


TR_TIME_PLANS_DIR    := build/googlenet_transition_plans
PLANS_GPU := $(foreach trans,$(TRANSITIONS),$(TR_TIME_PLANS_DIR)/googlenet_gpu_transition_at_$(trans).plan)
PLANS_DLA := $(foreach trans,$(TRANSITIONS),$(TR_TIME_PLANS_DIR)/googlenet_dla_transition_at_$(trans).plan)


TR_TIME_PROFILES_DIR := $(TR_TIME_PLANS_DIR)/profiles
TR_TIME_PROF_LOGS_DIR := $(TR_TIME_PLANS_DIR)/profile_logs



$(TR_TIME_PLANS_DIR)/googlenet_gpu_transition_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --starts_gpu True --output $@ --transition $* --verbose

$(TR_TIME_PLANS_DIR)/googlenet_dla_transition_at_%.plan:
	python3 src/build_engine.py --prototxt $(PROTOTXT) --starts_gpu False --output $@ --transition $* --verbose

$(TR_TIME_PROFILES_DIR)/%.profile: $(TR_TIME_PLANS_DIR)/%.plan
	mkdir -p $(TR_TIME_PROFILES_DIR) $(TR_TIME_PROF_LOGS_DIR)
	# The corresponding plan is found for each target profile here
	log_file=$(patsubst $(TR_TIME_PROFILES_DIR)/%.profile,$(TR_TIME_PROF_LOGS_DIR)/%.log,$@); \
	/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile \
	--exportProfile=$@ --avgRuns=1 --warmUp=5000 --duration=0 --loadEngine=$< > $$log_file

.PHONY: profiles
profiles: $(PLANS_GPU) $(PLANS_DLA)
	$(MAKE) $(patsubst $(TR_TIME_PLANS_DIR)/%.plan,$(TR_TIME_PROFILES_DIR)/%.profile,$^)


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
	mkdir -p output
	sudo python3 scripts/emc_analysis/emc_util_all.py

.PHONY: emc
emc: output/emc_results.json

all: emc profiles

clean:
	rm -rf output/* build/*

.PHONY: all clean
