# Step 1: Update paths as needed in the config.sh file, and choose between 'test' mode and 'final' mode
source ./config.sh

# Step 2: Create and activate the conda env
conda env create -f self_driving_env.yml
source activate self_driving_env

# Step 3: Download models
# Tranception (zero-shot)
mkdir $data_location/Tranception
curl -o $data_location/Tranception/Tranception_Large_checkpoint.zip https://marks.hms.harvard.edu/tranception/Tranception_Large_checkpoint.zip
unzip $data_location/Tranception/Tranception_Large_checkpoint.zip -d $data_location/Tranception && rm -f $data_location/Tranception/Tranception_Large_checkpoint.zip

# ESM2 (sequence embeddings)
mkdir $data_location/ESM
mkdir $data_location/ESM/ESM2
curl -o $data_location/ESM/ESM2/esm2_t33_650M_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
curl -o $data_location/ESM/ESM2/esm2_t33_650M_UR50D-contact-regression.pt https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt

# Step 4: Precompute zero-shot fitness predictions and sequence embeddings
## a. Generate all possible sequence combinations of provided segments via get_all_segment_combinations.sh
## b. Precompute zero-shot fitness scores for all these sequences via precompute_sero_shot.sh
## c. Precompute sequence embeddings for all these sequences via precompute_embeddings.sh
## Note: depending on the number of possible sequence combinations, these precomputing steps may be relatively time consuming, thereby defeating the purpose of precomputation.

# Step 5: Run self_driving.sh to perform Bayesian Optimization of starting sequences
## -- self_driving_single.sh --> script for single agent 
## -- self_driving_multiple.sh --> script for 3 agents running in parallel