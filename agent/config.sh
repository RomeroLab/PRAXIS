#!/bin/bash

# Overall config file for self driving experiment

# Set the run_mode to "test" for testing/debugging. Any other value (eg., 'final') will use the final run settings
export run_mode="final"

# Run ID = unique text string to identify the run (should be common to all 3 agents)
export run_id="Bgl_spec_2026"
# --- Resolve repository root (the agent/ directory) if not already provided ---
if [ -z "${PRAXIS_ROOT:-}" ]; then
  _praxis_d="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  while [ "$_praxis_d" != "/" ] && [ ! -f "$_praxis_d/self_driving.py" ]; do
    _praxis_d="$( dirname "$_praxis_d" )"
  done
  export PRAXIS_ROOT="$_praxis_d"
fi
# Path to main data location
export data_location=${PRAXIS_ROOT}
# Path to config of main model
export model_config_location="./configs/model/PNPT_ESM2_650M_final.json"
# Model type used for sequence embeddings
export model_type="ESM2_650M"
# Path to model used to get sequence embeddings
export model_location_sequence_embeddings=$data_location"/ESM/ESM2/esm2_t33_650M_UR50D.pt"
# Path to model used for zero-shot fitness predictions (Tranception)
export model_location_zero_shot=$data_location"/Tranception/Tranception_Large"
# Path to folder where we store all zero-shot fitness predictions
export zero_shot_fitness_predictions_folder=$data_location"/data/zero_shot_fitness_predictions/self_driving/ESM2_650M"
# Path to folder in which to score the embeddings for trained models
export model_checkpoint_folder=$data_location"/model_checkpoints/self_driving"
# Path to location where underlying database should be stored
export tablespace_location=$data_location"/data/tablespace/self_driving"
# Name of database with assayed sequences measurements to be shared across all agents
export database_name=$run_id"_"$run_mode"_database.db"
# Name of table with assayed sequences measurements to be shared across all agents
export table_name=$run_id"_"$run_mode"_assayed_sequences"
# Activity threshold value beyond which sequence is deemed active
export activity_threshold=0.001
# Whether to score all (remaining) combinations at each iteration, or sample (coditionally or at random)
export sampling_mode="conditional_sampling" #"all_combinations"
# Acquisition batch size during BO
export acquisition_batch=2
# ESM location (Path to ESM2 (650M) location -- used to get sequence embeddings used in clustering of acquisition function)
export ESM_location=$data_location"/ESM/ESM2/esm2_t33_650M_UR50D.pt"
# Path to config file that provides a mapping between target name and lab data acquisition index
export target_to_index_mapping="./configs/targets/target_name_to_lab_index.json"

# Resume settings (leave empty / 0 for normal start)
export seed_sequences="data/plate_data/phenotype.json"      # Path to CSV with pre-measured sequences to seed the DB
export starting_round=24        # BO round to resume from (0 = normal start)

if [ "$run_mode" = "test" ]; then
    ################################################################################################
    ### If testing codebase
    ################################################################################################
    #Path to csv file that contains sequence segments
    export sequence_segments_location='./data/sequence_segments.csv'
    #Path to location where we store a csv w/ all possible sequence segments combinations
    export all_sequences_location=$data_location"/data/all_sequences/GH1_all_sequences.csv"
    # Path to location where we store all precomputed sequence embeddings
    export embeddings_location=$data_location"/data/embeddings/self_driving/ESM2_650M/GH1_all_sequences_TEST.h5"
    # Path to location where we store all zero-shot fitness predictions
    export zero_shot_fitness_predictions_location=$zero_shot_fitness_predictions_folder"/GH1_all_sequences.csv"
    # Default starting sequence (GH1)
    export starting_sequence="MHHHHHHMVPAAQQTAMAPDAALTFPEGFLWGSATASYQIEGAAAEDGRTPSIWDTYARTPGRVRNGDTGDVATDHYHRWREDVALMAELGLGAYRFSLAWPRIQPTGRGPALQKGLDFYRRLADELLAKGIQPVATLYHWDLPQELENAGGWPERATAERFAEYAAIAADALGDRVKTWTTLNEPWCSAFLGYGSGVHAPGRTDPVAALRAAHHLNLGHGLAVQALRDRLPADAQCSVTLNIHHVRPLTDSDADADAVRRIDALANRVFTGPMLQGAYPEDLVKDTAGLTDWSFVRDGDLRLAHQKLDFLGVNYYSPTLVSEADGSGTHNSDGHGRSAHSPWPGADRVAFHQPPGETTAMGWAVDPSGLYELLRRLSSDFPALPLVITENGAAFHDYADPEGNVNDPERIAYVRDHLAAVHRAIKDGSDVRGYFLWSLLDNFEWAHGYSKRFGAVYVDYPTGTRIPKASARWYAEVARTGVLPTA"
    # Test parameter values
    export num_total_training_steps=10 #Number of total training steps
    export num_acquisitions=3 #Total number of data acquisitions (ie., number of Bayesian Optimization rounds)
    export num_MC_dropout_samples=2 #Number of MC dropout samples for uncertainty estimate
    export num_samples_per_iteration=100 # Number of samples generated at each iteation (if sampling_mode is "conditional_sampling" or "sample_at_random")
else
    ################################################################################################
    ### If using codebase for final runs
    ################################################################################################
    #Path to csv file that contains sequence segments
    export sequence_segments_location='./data/sequence_segments.csv'
    #Path to location where we store a csv w/ all possible sequence segments combinations
    export all_sequences_location=$data_location"/data/all_sequences/GH1_all_sequences.csv"
    # Path to location where we store all precomputed sequence embeddings
    export embeddings_location=$data_location"/data/embeddings/self_driving/ESM2_650M/GH1_all_sequences.h5"
    # Path to location where we store all zero-shot fitness predictions
    export zero_shot_fitness_predictions_location=$zero_shot_fitness_predictions_folder"/GH1_all_sequences.csv"
    # Default starting sequence (GH1)
    export starting_sequence="MHHHHHHMVPAAQQTAMAPDAALTFPEGFLWGSATASYQIEGAAAEDGRTPSIWDTYARTPGRVRNGDTGDVATDHYHRWREDVALMAELGLGAYRFSLAWPRIQPTGRGPALQKGLDFYRRLADELLAKGIQPVATLYHWDLPQELENAGGWPERATAERFAEYAAIAADALGDRVKTWTTLNEPWCSAFLGYGSGVHAPGRTDPVAALRAAHHLNLGHGLAVQALRDRLPADAQCSVTLNIHHVRPLTDSDADADAVRRIDALANRVFTGPMLQGAYPEDLVKDTAGLTDWSFVRDGDLRLAHQKLDFLGVNYYSPTLVSEADGSGTHNSDGHGRSAHSPWPGADRVAFHQPPGETTAMGWAVDPSGLYELLRRLSSDFPALPLVITENGAAFHDYADPEGNVNDPERIAYVRDHLAAVHRAIKDGSDVRGYFLWSLLDNFEWAHGYSKRFGAVYVDYPTGTRIPKASARWYAEVARTGVLPTA"
    # Test parameter values
    export num_total_training_steps=5000 #Number of total training steps
    export num_acquisitions=100 #Total number of data acquisitions (ie., number of Bayesian Optimization rounds)
    export num_MC_dropout_samples=5 #Number of MC dropout samples for uncertainty estimate
    export num_samples_per_iteration=1000 # Number of samples generated at each iteation (if sampling_mode is "conditional_sampling" or "sample_at_random")
fi