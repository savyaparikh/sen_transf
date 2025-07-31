import os
# dq_analysis_core_enhanced.py
# Enhanced version with better support for training data

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.linear_model import LogisticRegression
import sys
import joblib  # For saving/loading trained models
import numpy as np

# CONFIGURATION - Default paths
DEFAULT_INSTANCE_FILE_PATH = "C:/Users/savya/OneDrive/Desktop/Process Automation/DQ Check Instance.xlsx"
DEFAULT_TRAINING_DATA_PATH = "C:/Users/savya/OneDrive/Desktop/Process Automation/DQ_Training_Data.xlsx"
DEFAULT_MODEL_SAVE_PATH = "C:/Users/savya/OneDrive/Desktop/Process Automation/trained_models/"

def load_training_data(training_file_path):
    """Load and validate training data from Excel file"""
    try:
        # Load all sheets from the training data file
        training_data = {}
        
        # Sheet 1: DQ Nature Training
        if 'DQ_Nature_Training' in pd.ExcelFile(training_file_path).sheet_names:
            training_data['dq_nature'] = pd.read_excel(training_file_path, sheet_name='DQ_Nature_Training')
            
        # Sheet 2: Instance Name Training
        if 'Instance_Name_Training' in pd.ExcelFile(training_file_path).sheet_names:
            training_data['instance_name'] = pd.read_excel(training_file_path, sheet_name='Instance_Name_Training')
            
        # Sheet 3: FRASA Training
        if 'FRASA_Training' in pd.ExcelFile(training_file_path).sheet_names:
            training_data['frasa'] = pd.read_excel(training_file_path, sheet_name='FRASA_Training')
            
        # Sheet 4: Type Alignment Training
        if 'Type_Alignment_Training' in pd.ExcelFile(training_file_path).sheet_names:
            training_data['type_alignment'] = pd.read_excel(training_file_path, sheet_name='Type_Alignment_Training')
            
        return training_data
    except Exception as e:
        print(f"Warning: Could not load training data: {str(e)}")
        return {}

def save_trained_models(models, save_path):
    """Save trained models to disk for reuse"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for model_name, model in models.items():
        joblib.dump(model, os.path.join(save_path, f"{model_name}.pkl"))

def load_trained_models(load_path):
    """Load previously trained models from disk"""
    models = {}
    if os.path.exists(load_path):
        for file in os.listdir(load_path):
            if file.endswith('.pkl'):
                model_name = file.replace('.pkl', '')
                models[model_name] = joblib.load(os.path.join(load_path, file))
    return models

def run_analysis(main_file_path, instance_file_path=None, 
                training_file_path=None, use_saved_models=True,
                run_spellcheck=True, run_nlp_analysis=True, run_validation=True,
                progress_callback=None, log_callback=None):
    """
    Main function to run the DQ analysis with file paths as parameters
    
    Args:
        main_file_path: Path to the main Excel file
        instance_file_path: Path to instance file (uses DEFAULT_INSTANCE_FILE_PATH if None)
        training_file_path: Path to training data file (uses DEFAULT_TRAINING_DATA_PATH if None)
        use_saved_models: Whether to use previously saved models if available
        run_spellcheck: Whether to run spell check
        run_nlp_analysis: Whether to run NLP analysis
        run_validation: Whether to run validation checks
        progress_callback: Callback function for progress updates
        log_callback: Callback function for log messages
    """
    
    # Use default paths if not provided
    if instance_file_path is None:
        instance_file_path = DEFAULT_INSTANCE_FILE_PATH
    if training_file_path is None:
        training_file_path = DEFAULT_TRAINING_DATA_PATH
    
    def update_progress(value, status=""):
        if progress_callback:
            progress_callback(value, status)
    
    def log_message(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    try:
        # Load training data if available
        training_data = {}
        if training_file_path and os.path.exists(training_file_path):
            log_message(f"Loading training data from: {training_file_path}")
            training_data = load_training_data(training_file_path)
            log_message(f"Loaded training data for: {list(training_data.keys())}")
        
        # Check for saved models
        saved_models = {}
        if use_saved_models and os.path.exists(DEFAULT_MODEL_SAVE_PATH):
            saved_models = load_trained_models(DEFAULT_MODEL_SAVE_PATH)
            log_message(f"Loaded {len(saved_models)} saved models")
        
        def train_dq_nature_model(definitions: dict, training_df=None):
            # Check if we have a saved model
            if 'dq_nature_model' in saved_models and training_df is None:
                return saved_models['dq_nature_model']
            
            texts, labels = [], []
            for label, text in definitions.items():
                texts.append(text)
                labels.append(label)
            
            # Add training data if available
            if training_df is not None and not training_df.empty:
                valid_data = training_df.dropna(subset=["Activity (DQ Check Description)", "DQ Check Nature"])
                texts.extend(valid_data["Activity (DQ Check Description)"].tolist())
                labels.extend(valid_data["DQ Check Nature"].tolist())
                log_message(f"Added {len(valid_data)} training examples for DQ Nature model")
            
            pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
                ("classifier", MultinomialNB(alpha=0.1))
            ])
            pipeline.fit(texts, labels)
            return pipeline

        def predict_with_confidence(model, descriptions):
            proba = model.predict_proba(descriptions)
            preds = model.predict(descriptions)
            confidences = proba.max(axis=1)
            return preds, confidences

        update_progress(25, "Training DQ nature model...")
        log_message("Training DQ nature model...")
        
        definitions = {
            "Detective": "DQ checks are back-end checks designed to identify when an error, irregularity, or other undesirable event has occurred",
            "Preventive": "DQ checks are front-end checks designed to avoid the occurrence of errors, irregularities, or other undesirable events"
        }
        
        # Use training data if available
        dq_nature_training = training_data.get('dq_nature', pd.DataFrame())
        model = train_dq_nature_model(definitions, training_df=dq_nature_training)

        update_progress(30, "Loading main data file...")
        log_message("Loading main data file...")
        
        df = pd.read_excel(
            main_file_path, 
            sheet_name="Data Asset Level DQ Checks",
            header=8, usecols="C:Y", skiprows=range(9, 11))

        descs = df["Activity (DQ Check Description)"].fillna("").tolist()
        preds, confidences = predict_with_confidence(model, descs)
        df["Predicted_DQ_Check_Nature"] = preds
        df["Confidence"] = (confidences * 100).round(2)

        update_progress(35, "Loading instance data...")
        log_message(f"Loading instance data from: {instance_file_path}")
        
        if not os.path.exists(instance_file_path):
            log_message(f"Warning: Instance file not found at {instance_file_path}. Creating dummy data.")
            instance_df = pd.DataFrame({
                "DQ Check Instance Name": ["Sample Instance 1", "Sample Instance 2"],
                "Functional Description of the DQ Check": ["Sample Description 1", "Sample Description 2"],
                "Type of DQ Check": ["Detective", "Preventive"],
                "DQ Check Dimensions": ["Completeness", "Accuracy"]
            })
        else:
            instance_df = pd.read_excel(instance_file_path)
            log_message(f"Successfully loaded instance file with {len(instance_df)} records")
            
            # Validate required columns
            required_cols = ["DQ Check Instance Name", "Functional Description of the DQ Check", "Type of DQ Check"]
            missing_cols = [col for col in required_cols if col not in instance_df.columns]
            if missing_cols:
                log_message(f"Warning: Missing required columns in instance file: {missing_cols}")
            
            # Log unique Type of DQ Check values
            if "Type of DQ Check" in instance_df.columns:
                unique_types = instance_df["Type of DQ Check"].dropna().unique()
                log_message(f"Found {len(unique_types)} unique 'Type of DQ Check' values: {sorted(unique_types)[:5]}...")
                
                # Check if types match expected values
                expected_types = {"Detective", "Preventive", "Format", "Completeness", "Accuracy", 
                                 "Timeliness", "Uniqueness", "Consistency", "Reasonableness", "Integrity", "Other"}
                actual_types = set(unique_types)
                if not actual_types.intersection(expected_types):
                    log_message("Note: Instance file uses different type categories than expected.")
                    log_message(f"Expected types like: {sorted(list(expected_types))[:5]}...")
                    log_message(f"But found types like: {sorted(list(actual_types))[:5]}...")
        
        # Create combined text for better matching
        # Check which columns actually exist in the instance file
        combined_parts = []
        if "DQ Check Instance Name" in instance_df.columns:
            combined_parts.append(instance_df["DQ Check Instance Name"].fillna(''))
        if "Functional Description of the DQ Check" in instance_df.columns:
            combined_parts.append(instance_df["Functional Description of the DQ Check"].fillna(''))
        if "Type of DQ Check" in instance_df.columns:
            combined_parts.append(instance_df["Type of DQ Check"].fillna(''))
        if "DQ Check Dimensions" in instance_df.columns:
            combined_parts.append(instance_df["DQ Check Dimensions"].fillna(''))
        
        # Combine all available parts
        if combined_parts:
            instance_df["combined_text"] = combined_parts[0]
            for part in combined_parts[1:]:
                instance_df["combined_text"] = instance_df["combined_text"] + " " + part
        else:
            instance_df["combined_text"] = ""
        instance_names = instance_df["DQ Check Instance Name"].tolist()
        instance_texts = instance_df["combined_text"].tolist()

        update_progress(40, "Computing similarities...")
        log_message("Computing text similarities...")
        
        # --- Improved Top3 Instance Matching ---
        use_sentence_transformers = False
        try:
            from sentence_transformers import SentenceTransformer, util as st_util
            # Path to the local model directory
            local_model_path = r"C:\Users\savya\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots"
            # Find the latest snapshot directory
            snapshot_folders = [os.path.join(local_model_path, d) for d in os.listdir(local_model_path) if os.path.isdir(os.path.join(local_model_path, d))]
            if not snapshot_folders:
                raise RuntimeError("No snapshot folders found in the local model path.")
            latest_snapshot = max(snapshot_folders, key=os.path.getmtime)
            model_st = SentenceTransformer(latest_snapshot)
            use_sentence_transformers = True
            log_message(f"Using local Sentence Transformers model from: {latest_snapshot}")
        except ImportError:
            log_message("Sentence Transformers not available, falling back to TF-IDF.")
        except Exception as e:
            log_message(f"Could not load local Sentence Transformers model: {e}. Falling back to TF-IDF.")

        descs = df["Activity (DQ Check Description)"].fillna("").tolist()
        top3_instance_names = []
        top3_confidences = []
        
        if use_sentence_transformers:
            # Compute embeddings for all instance profiles and descriptions
            instance_embs = model_st.encode(instance_texts, convert_to_tensor=True, show_progress_bar=False)
            desc_embs = model_st.encode(descs, convert_to_tensor=True, show_progress_bar=False)
            for desc_emb in desc_embs:
                sims = st_util.pytorch_cos_sim(desc_emb, instance_embs).cpu().numpy().flatten()
                n_instances = len(instance_names)
                top_k = min(3, n_instances)
                top_idx = sims.argsort()[-top_k:][::-1] if n_instances > 0 else []
                names = [instance_names[i] for i in top_idx]
                confs = [round(float(sims[i])*100, 2) for i in top_idx]
                # Pad to length 3
                while len(names) < 3:
                    names.append("")
                    confs.append(0.0)
                top3_instance_names.append(names)
                top3_confidences.append(confs)
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            all_texts = descs + instance_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            desc_vectors = tfidf_matrix[:len(descs)]
            instance_vectors = tfidf_matrix[len(descs):]
            n_instances = len(instance_names)
            for desc_vec in desc_vectors:
                sims = cosine_similarity(desc_vec, instance_vectors).flatten() if n_instances > 0 else []
                top_k = min(3, n_instances)
                top_idx = sims.argsort()[-top_k:][::-1] if n_instances > 0 else []
                names = [instance_names[i] for i in top_idx]
                confs = [round(sims[i]*100, 2) for i in top_idx]
                # Pad to length 3
                while len(names) < 3:
                    names.append("")
                    confs.append(0.0)
                top3_instance_names.append(names)
                top3_confidences.append(confs)

        # Define pad_list function here before using it
        def pad_list(lst, length=3, pad_value=""):
            """Ensure a list has exactly 'length' elements by padding with pad_value"""
            if isinstance(lst, (list, tuple, np.ndarray)):
                lst = list(lst)
            else:
                lst = [lst]
            
            # Ensure we have exactly 'length' elements
            if len(lst) < length:
                lst = lst + [pad_value] * (length - len(lst))
            elif len(lst) > length:
                lst = lst[:length]
            
            return lst

        # Defensive fix: ensure every entry is a list of length 3
        # Add debugging to identify the issue
        try:
            # First, let's make sure top3_instance_names and top3_confidences are lists of lists
            fixed_names = []
            fixed_confs = []
            
            for i, (names, confs) in enumerate(zip(top3_instance_names, top3_confidences)):
                # Debug logging
                if i < 5:  # Log first 5 entries for debugging
                    log_message(f"Entry {i} - Names type: {type(names)}, Confs type: {type(confs)}")
                    log_message(f"Entry {i} - Names: {names}, Confs: {confs}")
                
                # Ensure names is a list
                if isinstance(names, (list, tuple, np.ndarray)):
                    names_list = list(names)
                elif isinstance(names, str):
                    names_list = [names] if names else []
                elif names is None or (isinstance(names, float) and np.isnan(names)):
                    names_list = []
                else:
                    names_list = [str(names)]
                
                # Ensure confs is a list
                if isinstance(confs, (list, tuple, np.ndarray)):
                    confs_list = list(confs)
                elif isinstance(confs, (int, float)):
                    confs_list = [float(confs)] if not (isinstance(confs, float) and np.isnan(confs)) else []
                elif confs is None:
                    confs_list = []
                else:
                    confs_list = [float(confs)]
                
                # Pad to length 3
                fixed_names.append(pad_list(names_list, 3, ""))
                fixed_confs.append(pad_list(confs_list, 3, 0.0))
            
            top3_instance_names = fixed_names
            top3_confidences = fixed_confs
            
        except Exception as e:
            log_message(f"Error in fixing top3 lists: {str(e)}")
            log_message(f"top3_instance_names sample: {top3_instance_names[:3] if top3_instance_names else 'empty'}")
            log_message(f"top3_confidences sample: {top3_confidences[:3] if top3_confidences else 'empty'}")
            raise

        # Now safely extract the values with additional error handling
        try:
            df["Top1_DQ_Check_Instance_Name"] = []
            df["Top1_Instance_Confidence"] = []
            df["Top2_DQ_Check_Instance_Name"] = []
            df["Top2_Instance_Confidence"] = []
            df["Top3_DQ_Check_Instance_Name"] = []
            df["Top3_Instance_Confidence"] = []
            
            for names, confs in zip(top3_instance_names, top3_confidences):
                # Names should now be guaranteed to be lists of length 3
                df["Top1_DQ_Check_Instance_Name"].append(names[0] if len(names) > 0 else "")
                df["Top1_Instance_Confidence"].append(confs[0] if len(confs) > 0 else 0.0)
                df["Top2_DQ_Check_Instance_Name"].append(names[1] if len(names) > 1 else "")
                df["Top2_Instance_Confidence"].append(confs[1] if len(confs) > 1 else 0.0)
                df["Top3_DQ_Check_Instance_Name"].append(names[2] if len(names) > 2 else "")
                df["Top3_Instance_Confidence"].append(confs[2] if len(confs) > 2 else 0.0)
                
        except Exception as e:
            log_message(f"Error extracting top3 values: {str(e)}")
            log_message(f"Current names: {names if 'names' in locals() else 'undefined'}")
            log_message(f"Current confs: {confs if 'confs' in locals() else 'undefined'}")
            raise

        df["Top3_DQ_Check_Instance_Names_Combined"] = (
            df["Top1_DQ_Check_Instance_Name"].fillna('') + " | " +
            df["Top2_DQ_Check_Instance_Name"].fillna('') + " | " +
            df["Top3_DQ_Check_Instance_Name"].fillna('')
        )
        df["Top3_Instance_Confidences_Combined"] = (
            df["Top1_Instance_Confidence"].astype(str).fillna('') + " | " +
            df["Top2_Instance_Confidence"].astype(str).fillna('') + " | " +
            df["Top3_Instance_Confidence"].astype(str).fillna('')
        )

        update_progress(50, "Training instance model...")
        log_message("Training instance name model...")
        
        def train_instance_name_model(instance_df, training_df=None):
            if 'instance_name_model' in saved_models and training_df is None:
                return saved_models['instance_name_model']
            
            # Preprocess text data
            def preprocess_text(text):
                text = str(text).lower()
                # Remove special characters but keep spaces
                text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
                # Remove extra whitespace
                text = ' '.join(text.split())
                return text
            
            texts = [preprocess_text(text) for text in instance_df["combined_text"].fillna("").tolist()]
            labels = instance_df["DQ Check Instance Name"].tolist()
            
            # Add training data if available
            if training_df is not None and not training_df.empty:
                valid_data = training_df.dropna(subset=["Activity (DQ Check Description)", "DQ Check Instance Name"])
                processed_texts = [preprocess_text(text) for text in valid_data["Activity (DQ Check Description)"].tolist()]
                texts.extend(processed_texts)
                labels.extend(valid_data["DQ Check Instance Name"].tolist())
                log_message(f"Added {len(valid_data)} training examples for Instance Name model")
            
            # Calculate class weights to handle imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            unique_labels = np.unique(labels)
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
            class_weight_dict = dict(zip(unique_labels, class_weights))
            
            # Create pipeline with enhanced features
            pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(
                    ngram_range=(1, 3),  # Include trigrams
                    max_features=10000,   # Increase features
                    min_df=2,            # Remove rare terms
                    max_df=0.95,         # Remove very common terms
                    sublinear_tf=True    # Apply sublinear scaling
                )),
                ("classifier", LogisticRegression(  # Switch to LogisticRegression
                    max_iter=1000,
                    class_weight=class_weight_dict,
                    multi_class='multinomial',
                    solver='lbfgs',
                    C=1.0                # Regularization parameter
                ))
            ])
            
            # Fit the pipeline
            pipeline.fit(texts, labels)
            
            return pipeline
        
        # Additional enhancement for semantic similarity matching
        def create_semantic_instance_matcher(instance_df, training_df=None):
            """
            Create a semantic matcher that uses embeddings for better instance matching
            This can be used alongside or instead of the ML model for better accuracy
            """
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Create instance profiles with rich features
            instance_profiles = {}
            
            for idx, row in instance_df.iterrows():
                instance_name = row["DQ Check Instance Name"]
                
                # Create a comprehensive profile for each instance
                profile_parts = []
                
                # Instance name (often contains key information)
                profile_parts.append(str(instance_name))
                
                # Functional description
                if "Functional Description of the DQ Check" in row and pd.notna(row["Functional Description of the DQ Check"]):
                    profile_parts.append(str(row["Functional Description of the DQ Check"]))
                
                # Type of check
                if "Type of DQ Check" in row and pd.notna(row["Type of DQ Check"]):
                    profile_parts.append(f"type: {row['Type of DQ Check']}")
                
                # Dimensions
                if "DQ Check Dimensions" in row and pd.notna(row["DQ Check Dimensions"]):
                    profile_parts.append(f"dimension: {row['DQ Check Dimensions']}")
                
                # Extract semantic keywords from instance name
                keywords = []
                
                # Data quality dimension keywords
                quality_dimensions = {
                    'completeness': ['complete', 'missing', 'null', 'blank', 'empty', 'populated', 'required'],
                    'accuracy': ['accurate', 'correct', 'valid', 'precise', 'exact', 'right'],
                    'consistency': ['consistent', 'match', 'align', 'agree', 'same', 'uniform'],
                    'timeliness': ['timely', 'late', 'delay', 'on-time', 'schedule', 'sla', 'deadline'],
                    'uniqueness': ['unique', 'duplicate', 'distinct', 'redundant', 'copy'],
                    'validity': ['valid', 'format', 'pattern', 'structure', 'syntax', 'rule'],
                    'integrity': ['integrity', 'referential', 'foreign', 'key', 'relationship', 'constraint']
                }
                
                for dimension, dim_keywords in quality_dimensions.items():
                    for keyword in dim_keywords:
                        if keyword in instance_name.lower() or (len(profile_parts) > 1 and keyword in ' '.join(profile_parts).lower()):
                            keywords.append(dimension)
                            keywords.extend(dim_keywords[:3])  # Add top related keywords
                            break
                
                # Check type keywords
                check_types = {
                    'reconciliation': ['reconcile', 'balance', 'match', 'compare', 'difference', 'variance'],
                    'threshold': ['threshold', 'limit', 'range', 'minimum', 'maximum', 'boundary'],
                    'trend': ['trend', 'pattern', 'change', 'increase', 'decrease', 'growth'],
                    'aggregation': ['sum', 'count', 'average', 'total', 'aggregate', 'group'],
                    'business_rule': ['rule', 'logic', 'condition', 'requirement', 'policy', 'standard']
                }
                
                for check_type, type_keywords in check_types.items():
                    for keyword in type_keywords:
                        if keyword in instance_name.lower() or (len(profile_parts) > 1 and keyword in ' '.join(profile_parts).lower()):
                            keywords.append(check_type)
                            keywords.extend(type_keywords[:3])
                            break
                
                # Add keywords to profile
                if keywords:
                    profile_parts.append(f"keywords: {' '.join(set(keywords))}")
                
                # Add examples from training data if available
                if training_df is not None and not training_df.empty:
                    examples = training_df[training_df["DQ Check Instance Name"] == instance_name]["Activity (DQ Check Description)"].tolist()
                    if examples:
                        # Add up to 3 examples
                        profile_parts.append(f"examples: {' | '.join(examples[:3])}")
                
                instance_profiles[instance_name] = ' '.join(profile_parts).lower()
            
            # Create vectorizer with enhanced features
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                min_df=1,
                sublinear_tf=True,
                use_idf=True,
                token_pattern=r'(?u)\b\w+\b'  # Include single character tokens
            )
            
            # Fit vectorizer on all profiles
            profile_texts = list(instance_profiles.values())
            profile_vectors = vectorizer.fit_transform(profile_texts)
            instance_names = list(instance_profiles.keys())
            
            def match_instance(description, top_k=5, confidence_threshold=0.3):
                """
                Match a description to the most likely instance names
                Returns top K matches with confidence scores
                """
                # Preprocess description
                desc_lower = str(description).lower()
                
                # Extract features from description
                desc_features = [desc_lower]
                
                # Add detected dimensions and types
                for dimension, dim_keywords in quality_dimensions.items():
                    if any(keyword in desc_lower for keyword in dim_keywords):
                        desc_features.append(f"dimension: {dimension}")
                
                for check_type, type_keywords in check_types.items():
                    if any(keyword in desc_lower for keyword in type_keywords):
                        desc_features.append(f"type: {check_type}")
                
                # Vectorize description
                desc_text = ' '.join(desc_features)
                desc_vector = vectorizer.transform([desc_text])
                
                # Calculate similarities
                similarities = cosine_similarity(desc_vector, profile_vectors).flatten()
                
                # Get top K matches
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                matches = []
                for idx in top_indices:
                    if similarities[idx] >= confidence_threshold:
                        matches.append({
                            'instance_name': instance_names[idx],
                            'confidence': similarities[idx],
                            'profile': instance_profiles[instance_names[idx]][:200]  # First 200 chars of profile
                        })
                
                return matches
            
            return match_instance, instance_profiles


        # Enhanced prediction function that combines ML model with semantic matching
        def predict_instance_with_ensemble(description, ml_model, semantic_matcher, instance_profiles):
            """
            Ensemble prediction combining ML model and semantic matching
            """
            # Get ML model prediction
            ml_pred = ml_model.predict([description])[0]
            ml_proba = ml_model.predict_proba([description])[0]
            ml_confidence = ml_proba.max()
            
            # Get semantic matches
            semantic_matches = semantic_matcher(description, top_k=3)
            
            # Combine predictions
            if semantic_matches:
                top_semantic = semantic_matches[0]
                
                # If both agree, boost confidence
                if ml_pred == top_semantic['instance_name']:
                    final_prediction = ml_pred
                    final_confidence = min(0.95, (ml_confidence + top_semantic['confidence']) / 2 * 1.1)
                
                # If ML confidence is low but semantic is high, use semantic
                elif ml_confidence < 0.4 and top_semantic['confidence'] > 0.6:
                    final_prediction = top_semantic['instance_name']
                    final_confidence = top_semantic['confidence']
                
                # If ML confidence is high, trust it
                elif ml_confidence > 0.7:
                    final_prediction = ml_pred
                    final_confidence = ml_confidence
                
                # Otherwise, use weighted average
                else:
                    # Check if semantic match is in top 3 ML predictions
                    ml_top3_idx = ml_proba.argsort()[-3:][::-1]
                    ml_classes = ml_model.classes_
                    
                    if top_semantic['instance_name'] in [ml_classes[i] for i in ml_top3_idx]:
                        final_prediction = top_semantic['instance_name']
                        final_confidence = (ml_confidence + top_semantic['confidence']) / 2
                    else:
                        # Use ML prediction but with reduced confidence
                        final_prediction = ml_pred
                        final_confidence = ml_confidence * 0.8
            else:
                # No good semantic match, use ML prediction
                final_prediction = ml_pred
                final_confidence = ml_confidence
            
            return final_prediction, final_confidence


        # Modification to the main run_analysis function to use enhanced instance matching
        # Replace the instance prediction section with:
        
        # After training the instance model
        instance_name_training = training_data.get('instance_name', pd.DataFrame())
        instance_model = train_instance_name_model(instance_df, training_df=instance_name_training)

        # Create semantic matcher
        semantic_matcher, instance_profiles = create_semantic_instance_matcher(instance_df, instance_name_training)

        # Enhanced prediction with ensemble
        desc_texts = df["Activity (DQ Check Description)"].fillna("").tolist()
        instance_preds = []
        instance_confidences = []

        for desc in desc_texts:
            pred, conf = predict_instance_with_ensemble(desc, instance_model, semantic_matcher, instance_profiles)
            instance_preds.append(pred)
            instance_confidences.append(conf)

        df["Predicted_DQ_Check_Instance_Name"] = instance_preds
        df["Predicted_Instance_Confidence"] = (np.array(instance_confidences) * 100).round(2)

        update_progress(55, "Processing FRASA criteria...")
        log_message("Processing FRASA criteria...")

        
        # Enhanced FRASA with training data
        frasa_training = training_data.get('frasa', pd.DataFrame())
        frasa_keywords = {
            "Frequency": ["daily", "weekly", "monthly", "quarterly", "yearly", "once", "every", "periodic", "hourly", "real time", "real-time", "annually", "semi-annually"],
            "Responsible": ["by", "owner", "team", "user", "manager", "responsible", "party", "admin", "analyst", "performed by", "executed by", "managed by"],
            "Activity": ["check", "validate", "review", "monitor", "ensure", "verify", "activity", "compare", "reconcile", "audit", "inspect", "examine"],
            "Source": ["from", "source", "system", "file", "input", "table", "database", "report", "extract", "data from", "sourced from", "extracted from"],
            "Action": ["action", "flag", "notify", "report", "correct", "remediate", "taken", "alert", "escalate", "update", "email", "ticket", "log"]
        }
        
        frasa_models = {}
        if not frasa_training.empty:
            log_message("Training FRASA models with labeled data...")
            vectorizer_frasa = TfidfVectorizer(ngram_range=(1, 2))
            X = vectorizer_frasa.fit_transform(frasa_training["Description"].fillna(""))
            for crit in ["Frequency", "Responsible", "Activity", "Source", "Action"]:
                if crit in frasa_training.columns:
                    y = frasa_training[crit].astype(int)
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X, y)
                    frasa_models[crit] = (model, vectorizer_frasa)

        def check_frasa_criteria(description):
            missing = []
            desc = description.lower() if isinstance(description, str) else ""
            for crit, keywords in frasa_keywords.items():
                found = False
                if crit in frasa_models:
                    model, vect = frasa_models[crit]
                    pred = model.predict(vect.transform([desc]))[0]
                    found = bool(pred)
                else:
                    for word in keywords:
                        if re.search(r'\b' + re.escape(word) + r'\b', desc):
                            found = True
                            break
                if not found:
                    missing.append(crit)
            return missing

        def extract_frequency(text):
            freq_keywords = ["daily", "weekly", "monthly", "quarterly", "yearly", "once", "every", "periodic", "hourly", "real time", "real-time", "annually", "semi-annually"]
            for word in freq_keywords:
                if re.search(r'\b' + re.escape(word) + r'\b', text.lower()):
                    return word
            return ""

        def extract_source(text):
            match = re.search(r'(from|source|sourced from|extracted from)\s+([a-zA-Z0-9_ -]+)', text.lower())
            if match:
                return match.group(2).strip()
            return ""

        def extract_action(text):
            action_keywords = ["flag", "notify", "report", "correct", "remediate", "taken", "alert", "escalate", "update", "action", "email", "ticket", "log"]
            for word in action_keywords:
                if re.search(r'\b' + re.escape(word) + r'\b', text.lower()):
                    return word
            return ""

        def extract_automation(text):
            if re.search(r'automat(ed|ion|ically)|manual(ly)?', text.lower()):
                if "manual" in text.lower():
                    return "Manual"
                else:
                    return "Automated"
            return ""

        def nlp_source_match(desc, source):
            desc = str(desc).lower()
            source = str(source).lower()
            source_tokens = re.findall(r'\b\w+\b', source)
            for token in source_tokens:
                if token in desc:
                    return True
            vect = TfidfVectorizer().fit([desc, source])
            vecs = vect.transform([desc, source])
            sim = cosine_similarity(vecs[0], vecs[1])[0][0]
            return sim > 0.3

        def nlp_action_taken_match(desc, action_taken):
            desc = str(desc).lower()
            action_taken = str(action_taken).lower()
            if action_taken in desc:
                return True
            vect = TfidfVectorizer().fit([desc, action_taken])
            vecs = vect.transform([desc, action_taken])
            sim = cosine_similarity(vecs[0], vecs[1])[0][0]
            return sim > 0.3

        update_progress(60, "Parsing column data...")
        log_message("Parsing column data...")
        
        parse_test_rows = []
        for idx, row in df.iterrows():
            desc = str(row["Activity (DQ Check Description)"])
            parsed = {
                "Frequency": extract_frequency(desc),
                "DQ Check Source": extract_source(desc),
                "DQ Check Action Taken": extract_action(desc),
                "Extent of Automation": extract_automation(desc)
            }
            mismatches = []
            if "Frequency" in df.columns and parsed["Frequency"]:
                if str(row["Frequency"]).strip().lower() not in parsed["Frequency"]:
                    mismatches.append("Frequency")
            if "DQ Check Action Taken" in df.columns and parsed["DQ Check Action Taken"]:
                action_taken_val = str(row["DQ Check Action Taken"]).strip().lower()
                if not nlp_action_taken_match(desc, action_taken_val):
                    mismatches.append("Action Taken")
            if "Extent of Automation" in df.columns and parsed["Extent of Automation"]:
                if str(row["Extent of Automation"]).strip().lower() not in parsed["Extent of Automation"].lower():
                    mismatches.append("Extent of Automation")
            source = str(row["DQ Check Source"]) if "DQ Check Source" in df.columns else ""
            if source.strip():
                if not nlp_source_match(desc, source):
                    mismatches.append("Source")
            if mismatches:
                value = f"Mismatch in: {', '.join(mismatches)}"
                confidence = round((4 - len(mismatches)) / 4 * 100, 2)
            else:
                value = "All parsed values match"
                confidence = 100
            parse_test_rows.append({
                "Activity (DQ Check Description)": desc,
                "Type": "Parsed_Column_Check",
                "Value": value,
                "Confidence (%)": confidence
            })

        parse_test_df = pd.DataFrame(parse_test_rows)

        # Rest of the original code continues...
        # Create result rows for different analysis types
        nature_rows = df[["Activity (DQ Check Description)", "Predicted_DQ_Check_Nature", "Confidence"]].copy()
        nature_rows["Type"] = "Predicted_DQ_Check_Nature"
        nature_rows = nature_rows.rename(columns={
            "Predicted_DQ_Check_Nature": "Value",
            "Confidence": "Confidence (%)"
        })[["Activity (DQ Check Description)", "Type", "Value", "Confidence (%)"]]

        top3_combined_rows = df[["Activity (DQ Check Description)", "Top3_DQ_Check_Instance_Names_Combined", "Top3_Instance_Confidences_Combined"]].copy()
        top3_combined_rows["Type"] = "Top3_DQ_Check_Instance_Names_Combined"
        top3_combined_rows = top3_combined_rows.rename(columns={
            "Top3_DQ_Check_Instance_Names_Combined": "Value",
            "Top3_Instance_Confidences_Combined": "Confidence (%)"
        })[["Activity (DQ Check Description)", "Type", "Value", "Confidence (%)"]]

        pred_instance_rows = df[["Activity (DQ Check Description)", "Predicted_DQ_Check_Instance_Name", "Predicted_Instance_Confidence"]].copy()
        pred_instance_rows["Type"] = "Predicted_DQ_Check_Instance_Name"
        pred_instance_rows = pred_instance_rows.rename(columns={
            "Predicted_DQ_Check_Instance_Name": "Value",
            "Predicted_Instance_Confidence": "Confidence (%)"
        })[["Activity (DQ Check Description)", "Type", "Value", "Confidence (%)"]]

        update_progress(65, "Processing FRASA criteria...")
        
        frasa_rows = []
        for idx, desc in enumerate(df["Activity (DQ Check Description)"].fillna('')):
            missing = check_frasa_criteria(desc)
            total = 5
            present = total - len(missing)
            confidence = round((present / total) * 100, 2)
            if missing:
                value = f"Missing: {', '.join(missing)}"
            else:
                value = "All criteria present"
            frasa_rows.append({
                "Activity (DQ Check Description)": desc,
                "Type": "FRASA_Criteria_Check",
                "Value": value,
                "Confidence (%)": confidence
            })
        frasa_df = pd.DataFrame(frasa_rows)

        update_progress(70, "Running validation checks...")
        log_message("Running validation checks...")
        
        # Version check
        version_sheet = pd.read_excel(
            main_file_path,
            sheet_name="Data Asset Level DQ Checks",
            header=None,
            usecols="B",
            nrows=1
        )
        version_value = str(version_sheet.iloc[0, 0]).strip()

        version_test_row = {
            "Activity (DQ Check Description)": "Check if cell B1 in 'Data Asset Level DQ Checks' sheet is Version 5.2",
            "Type": "Version_Check",
            "Value": "Pass" if version_value == "Version 5.2" else f"Fail (Found: {version_value})",
            "Confidence (%)": 100.0 if version_value == "Version 5.2" else 0.0
        }

        # Asset sheet fill check
        asset_df = pd.read_excel(
            main_file_path,
            sheet_name="Data Asset Level DQ Checks",
            header=8,
            usecols="C:Y",
            skiprows=range(9, 11)
        )
        
        try:
            element_df = pd.read_excel(
                main_file_path,
                sheet_name="Data Element Level DQ Checks",
                header=8,
                usecols="C:Y",
                skiprows=range(9, 11)
            )
            element_filled = not element_df.dropna(how='all').empty
        except:
            element_filled = False

        asset_filled = not asset_df.dropna(how='all').empty

        if asset_filled:
            value = "Pass"
            confidence = 100.0
        elif element_filled:
            value = "Fail (Only 'Data Element Level DQ Checks' filled, 'Data Asset Level DQ Checks' is empty)"
            confidence = 0.0
        else:
            value = "Fail (No data in either sheet)"
            confidence = 0.0

        custom_fill_test_row = {
            "Activity (DQ Check Description)": "Check if 'Data Asset Level DQ Checks' sheet is filled from C12:Y12 onwards",
            "Type": "Asset_Sheet_Fill_Check",
            "Value": value,
            "Confidence (%)": confidence
        }

        # Data Source ID consistency check
        data_source_ids = df["Data Source ID"].dropna().astype(str).tolist() if "Data Source ID" in df.columns else []

        if len(data_source_ids) > 0 and all(x == data_source_ids[0] for x in data_source_ids):
            value = "Pass"
            confidence = 100.0
        else:
            value = f"Fail (Found IDs: {', '.join(sorted(set(data_source_ids)))})"
            confidence = 0.0

        data_source_id_test_row = {
            "Activity (DQ Check Description)": "Check if 'Data Source ID' values are consistent",
            "Type": "Data_Source_ID_Consistency_Check",
            "Value": value,
            "Confidence (%)": confidence
        }

        # EDCC check
        if "EDCC Monitoring Control Tracking ID" in df.columns:
            edcc_values = df["EDCC Monitoring Control Tracking ID"].dropna().astype(str)
            if not edcc_values.empty and all(val.strip().lower() == "unknown" for val in edcc_values):
                value = "Pass"
                confidence = 100.0
            else:
                non_unknowns = edcc_values[edcc_values.str.lower().str.strip() != "unknown"].unique()
                value = f"Fail (Found: {', '.join(non_unknowns)})"
                confidence = 0.0
        else:
            value = "Fail (Column not found)"
            confidence = 0.0

        edcc_test_row = {
            "Activity (DQ Check Description)": "Check if all 'EDCC Monitoring Control Tracking ID' values are 'Unknown'",
            "Type": "EDCC_Monitoring_Control_Tracking_ID_Check",
            "Value": value,
            "Confidence (%)": confidence
        }

        # Risk Instance Name check
        if "Risk Instance Name" in df.columns:
            risk_values = df["Risk Instance Name"].dropna().astype(str)
            if not risk_values.empty and all(val.strip().lower().startswith("risk of business execution") for val in risk_values):
                value = "Pass"
                confidence = 100.0
            else:
                non_matching = risk_values[~risk_values.str.lower().str.strip().str.startswith("risk of business execution")].unique()
                value = f"Fail (Found: {', '.join(non_matching)})"
                confidence = 0.0
        else:
            value = "Fail (Column not found)"
            confidence = 0.0

        risk_instance_test_row = {
            "Activity (DQ Check Description)": "Check if all 'Risk Instance Name' values start with 'Risk of business execution'",
            "Type": "Risk_Instance_Name_Check",
            "Value": value,
            "Confidence (%)": confidence
        }

        # Risk Event Level 3 Name check
        if "Risk Event Level 3 Name" in df.columns:
            risk_event_values = df["Risk Event Level 3 Name"].dropna().astype(str)
            if not risk_event_values.empty and all(val.strip().lower() == "inadequate data integrity/quality" for val in risk_event_values):
                value = "Pass"
                confidence = 100.0
            else:
                non_matching = risk_event_values[risk_event_values.str.lower().str.strip() != "inadequate data integrity/quality"].unique()
                value = f"Fail (Found: {', '.join(non_matching)})"
                confidence = 0.0
        else:
            value = "Fail (Column not found)"
            confidence = 0.0

        risk_event_test_row = {
            "Activity (DQ Check Description)": "Check if all 'Risk Event Level 3 Name' values are 'Inadequate Data Integrity/Quality'",
            "Type": "Risk_Event_Level_3_Name_Check",
            "Value": value,
            "Confidence (%)": confidence
        }

        # Control Type Level 3 Name check
        if "Control Type Level 3 Name" in df.columns:
            control_type_values = df["Control Type Level 3 Name"].dropna().astype(str)
            if not control_type_values.empty and all(val.strip().lower() == "data quality controls" for val in control_type_values):
                value = "Pass"
                confidence = 100.0
            else:
                non_matching = control_type_values[control_type_values.str.lower().str.strip() != "data quality controls"].unique()
                value = f"Fail (Found: {', '.join(non_matching)})"
                confidence = 0.0
        else:
            value = "Fail (Column not found)"
            confidence = 0.0

        control_type_test_row = {
            "Activity (DQ Check Description)": "Check if all 'Control Type Level 3 Name' values are 'Data Quality Controls'",
            "Type": "Control_Type_Level_3_Name_Check",
            "Value": value,
            "Confidence (%)": confidence
        }

        update_progress(75, "Running spell check...")
        log_message("Running spell check...")
        
        # Spell check with proper error handling
        spellcheck_test_row = {
            "Activity (DQ Check Description)": "Spell check on all text fields in 'Data Asset Level DQ Checks' sheet",
            "Type": "Spell_Check",
            "Value": "Skipped (pyspellchecker not available)",
            "Confidence (%)": 0.0
        }
        
        if run_spellcheck:
            try:
                from spellchecker import SpellChecker
                
                try:
                    spell = SpellChecker()
                    
                    misspelled_words = set()
                    for col in df.select_dtypes(include='object').columns:
                        for val in df[col].dropna():
                            words = str(val).split()
                            words = [w for w in words if len(w) > 2 and w.isalpha()]
                            misspelled = spell.unknown(words)
                            misspelled = [w for w in misspelled if not (w.isupper() and len(w) <= 4)]
                            misspelled_words.update(misspelled)

                    if not misspelled_words:
                        value = "Pass"
                        confidence = 100.0
                    else:
                        words_list = sorted(misspelled_words)[:10]
                        if len(misspelled_words) > 10:
                            value = f"Fail (Misspelled words: {', '.join(words_list)}, and {len(misspelled_words) - 10} more)"
                        else:
                            value = f"Fail (Misspelled words: {', '.join(words_list)})"
                        confidence = 0.0
                    
                    spellcheck_test_row = {
                        "Activity (DQ Check Description)": "Spell check on all text fields",
                        "Type": "Spell_Check",
                        "Value": value,
                        "Confidence (%)": confidence
                    }
                except Exception as e:
                    log_message(f"Spell checker initialization failed: {str(e)}. Using basic spell check.")
                    
                    common_misspellings = {
                        'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
                        'definately': 'definitely', 'occurance': 'occurrence', 'aquire': 'acquire',
                        'acheive': 'achieve', 'calender': 'calendar', 'collegue': 'colleague',
                        'concious': 'conscious', 'experiance': 'experience', 'independant': 'independent',
                        'neccessary': 'necessary', 'noticable': 'noticeable', 'occassion': 'occasion',
                        'paralell': 'parallel', 'priviledge': 'privilege', 'reccomend': 'recommend',
                        'rythm': 'rhythm', 'sieze': 'seize'
                    }
                    
                    found_misspellings = []
                    for col in df.select_dtypes(include='object').columns:
                        for val in df[col].dropna():
                            text = str(val).lower()
                            for misspelling in common_misspellings:
                                if misspelling in text:
                                    found_misspellings.append(misspelling)
                    
                    if found_misspellings:
                        value = f"Basic check found potential issues: {', '.join(set(found_misspellings)[:5])}"
                        confidence = 50.0
                    else:
                        value = "Basic spell check passed"
                        confidence = 90.0
                    
                    spellcheck_test_row = {
                        "Activity (DQ Check Description)": "Spell check on all text fields",
                        "Type": "Spell_Check",
                        "Value": value,
                        "Confidence (%)": confidence
                    }
                    
            except ImportError:
                log_message("Warning: pyspellchecker not available. Skipping spell check.")

        update_progress(80, "Processing type alignment...")
        log_message("Processing type alignment...")
        
        # Type alignment checks with training data support
        instance_names = instance_df["DQ Check Instance Name"].fillna("").tolist()
        vectorizer_align = TfidfVectorizer().fit(instance_names)
        best_instance_names = []
        for desc in df["Activity (DQ Check Description)"].fillna(""):
            desc_vec = vectorizer_align.transform([desc])
            sims = cosine_similarity(desc_vec, vectorizer_align.transform(instance_names)).flatten()
            idx = sims.argmax()
            best_instance_names.append(instance_names[idx])

        df["Best_Match_Instance_Name"] = best_instance_names

        def train_type_model_with_instance(df, training_df=None):
            if 'type_alignment_model' in saved_models and training_df is None:
                return saved_models['type_alignment_model']
                
            texts, labels = [], []
            
            # First try to get training data from the main dataframe
            # Only use rows where we have all necessary fields
            df_train = df.dropna(subset=["Activity (DQ Check Description)", "Type of DQ Check"])
            if "Best_Match_Instance_Name" in df_train.columns:
                df_train = df_train.dropna(subset=["Best_Match_Instance_Name"])
                combined = df_train["Activity (DQ Check Description)"].astype(str) + " " + df_train["Best_Match_Instance_Name"].astype(str)
                texts.extend(combined.tolist())
                labels.extend(df_train["Type of DQ Check"].tolist())
            else:
                # If Best_Match_Instance_Name not available, use just the description
                texts.extend(df_train["Activity (DQ Check Description)"].astype(str).tolist())
                labels.extend(df_train["Type of DQ Check"].tolist())
                
            # Add training data if available
            if training_df is not None and not training_df.empty:
                valid_data = training_df.dropna(subset=["Activity (DQ Check Description)", "Type of DQ Check"])
                texts.extend(valid_data["Activity (DQ Check Description)"].tolist())
                labels.extend(valid_data["Type of DQ Check"].tolist())
                log_message(f"Added {len(valid_data)} training examples for Type Alignment model")
                
            if not texts:
                log_message("No training data available for type model")
                return None
                
            # Filter out any empty texts or labels
            valid_pairs = [(t, l) for t, l in zip(texts, labels) if t.strip() and l.strip()]
            if not valid_pairs:
                return None
                
            texts, labels = zip(*valid_pairs)
            
            pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", MultinomialNB(alpha=0.1))
            ])
            pipeline.fit(texts, labels)
            return pipeline

        type_alignment_training = training_data.get('type_alignment', pd.DataFrame())
        type_model = train_type_model_with_instance(df, training_df=type_alignment_training)

        test_rows = []
        for idx, row in df.iterrows():
            desc = str(row["Activity (DQ Check Description)"])
            instance = str(row.get("Best_Match_Instance_Name", ""))
            combined = desc + " " + instance
            
            if type_model:
                try:
                    pred_type = type_model.predict([combined])[0]
                    # Get prediction probabilities for confidence
                    pred_proba = type_model.predict_proba([combined])[0]
                    pred_confidence = max(pred_proba) * 100
                except:
                    pred_type = "Unknown"
                    pred_confidence = 0.0
            else:
                pred_type = "Unknown"
                pred_confidence = 0.0
                
            actual_type = str(row.get("Type of DQ Check", ""))
            
            # Determine alignment and suggested type
            if actual_type == pred_type:
                suggested_type = pred_type
                value = f"Aligned - Type: {pred_type}"
                confidence = pred_confidence
            elif actual_type in ["", "nan", "None"]:
                suggested_type = pred_type
                value = f"Suggested Type: {pred_type}"
                confidence = pred_confidence
            else:
                suggested_type = actual_type  # Keep the actual type if different
                value = f"Mismatch - Actual: {actual_type}, Predicted: {pred_type}"
                confidence = 100.0 - pred_confidence  # Lower confidence for mismatches
                
            test_rows.append({
                "Activity (DQ Check Description)": desc,
                "Type": "Type_of_DQ_Check_Alignment",
                "Value": value,
                "Confidence (%)": round(confidence, 2)
            })

        type_align_df = pd.DataFrame(test_rows)

        # Instance Type Alignment Check
        instance_lookup = {}
        if "DQ Check Instance Name" in instance_df.columns and "Type of DQ Check" in instance_df.columns:
            instance_lookup = dict(zip(
                instance_df["DQ Check Instance Name"].astype(str).str.strip(),
                instance_df["Type of DQ Check"].astype(str).str.strip()
            ))
            log_message(f"Instance lookup created with {len(instance_lookup)} entries")
            # Log unique types found in instance file
            unique_types = instance_df["Type of DQ Check"].dropna().unique()
            log_message(f"Unique 'Type of DQ Check' values in instance file: {sorted(unique_types)}")

        alignment_results = []
        for idx, row in df.iterrows():
            instance_name = str(row.get("DQ Check Instance Name", "")).strip()
            dq_type = str(row.get("Type of DQ Check", "")).strip()
            ref_type = instance_lookup.get(instance_name)
            
            # Special handling for different type systems
            if dq_type == "Other":
                value = "Pass"
                confidence = 100.0
            elif ref_type is None:
                value = f"Fail (Instance '{instance_name}' not found in reference)"
                confidence = 0.0
            elif dq_type == ref_type:
                value = "Pass"
                confidence = 100.0
            else:
                # Check if this might be a valid mismatch due to different categorization systems
                value = f"Fail (Expected: {ref_type}, Found: {dq_type})"
                confidence = 0.0
                
            alignment_results.append({
                "Activity (DQ Check Description)": row.get("Activity (DQ Check Description)", ""),
                "Type": "Instance_Type_Alignment",
                "Value": value,
                "Confidence (%)": confidence
            })

        alignment_df = pd.DataFrame(alignment_results)

        # Manual evidence check
        manual_evidence_row = {
            "Activity (DQ Check Description)": "",
            "Type": "manual_evidence_check",
            "Value": "Evidence should be provided by the ABO to substantiate the primary data quality",
            "Confidence (%)": ""
        }

        update_progress(85, "Formatting final results...")
        log_message("Formatting final results...")
        
        # Save trained models if new training data was used
        if training_data and not use_saved_models:
            models_to_save = {}
            if 'dq_nature' in training_data:
                models_to_save['dq_nature_model'] = model
            if 'instance_name' in training_data:
                models_to_save['instance_name_model'] = instance_model
            if 'type_alignment' in training_data and type_model:
                models_to_save['type_alignment_model'] = type_model
            
            if models_to_save:
                save_trained_models(models_to_save, DEFAULT_MODEL_SAVE_PATH)
                log_message(f"Saved {len(models_to_save)} trained models")
        
        # Custom output formatting
        def add_custom_columns(df, nlp_types=None):
            if nlp_types is None:
                nlp_types = [
                    "Predicted_DQ_Check_Nature",
                    "Top3_DQ_Check_Instance_Names_Combined",
                    "Predicted_DQ_Check_Instance_Name",
                    "Type_of_DQ_Check_Alignment",
                    "Instance_Type_Alignment"
                ]
            
            if "Value" in df.columns:
                df = df.rename(columns={"Value": "results"})
            
            for col in [
                "Check#", "Check Description", "Action Required", "Link to Referenced Document",
                "Activity (DQ Check Description)", "results", "Type", "Confidence (%)", "Comments"
            ]:
                if col not in df.columns:
                    df[col] = ""
            
            df = df[[
                "Check#", "Check Description", "Action Required", "Link to Referenced Document",
                "Activity (DQ Check Description)", "results", "Type", "Confidence (%)", "Comments"
            ]]
            
            if "Type" in df.columns and "Comments" in df.columns:
                df.loc[df["Type"].isin(nlp_types), "Comments"] = "Note: AI generated response"
            
            return df

        # List of NLP/AI generated types
        nlp_types = [
            "Predicted_DQ_Check_Nature",
            "Top3_DQ_Check_Instance_Names_Combined",
            "Predicted_DQ_Check_Instance_Name",
            "Type_of_DQ_Check_Alignment",
            "Instance_Type_Alignment"
        ]

        # Apply formatting to all DataFrames
        nature_rows = add_custom_columns(nature_rows.copy(), nlp_types)
        top3_combined_rows = add_custom_columns(top3_combined_rows.copy(), nlp_types)
        pred_instance_rows = add_custom_columns(pred_instance_rows.copy(), nlp_types)
        frasa_df = add_custom_columns(frasa_df.copy(), nlp_types)
        parse_test_df = add_custom_columns(parse_test_df.copy(), nlp_types)
        type_align_df = add_custom_columns(type_align_df.copy(), nlp_types)
        alignment_df = add_custom_columns(alignment_df.copy(), nlp_types)

        # Concatenate all test cases
        dfs_to_concat = [
            nature_rows,
            top3_combined_rows,
            pred_instance_rows,
            frasa_df,
            parse_test_df,
            type_align_df,
            alignment_df
        ]

        long_df = pd.concat(dfs_to_concat, ignore_index=True)

        # Standardize and append single-row test cases
        def standardize_single_row(row):
            row = row.copy()
            row["results"] = row.get("Value", "")
            row["Activity (DQ Check Description)"] = row.get("Activity (DQ Check Description)", "")
            row.pop("Value", None)
            
            for col in [
                "Check#", "Check Description", "Action Required", "Link to Referenced Document",
                "results", "Type", "Confidence (%)", "Comments", "Activity (DQ Check Description)"
            ]:
                if col not in row:
                    row[col] = ""
            
            ordered = [
                "Check#", "Check Description", "Action Required", "Link to Referenced Document",
                "Activity (DQ Check Description)", "results", "Type", "Confidence (%)", "Comments"
            ]
            return {k: row[k] for k in ordered}

        # List all single-row test cases
        single_row_tests = [
            version_test_row,
            custom_fill_test_row,
            data_source_id_test_row,
            edcc_test_row,
            risk_instance_test_row,
            risk_event_test_row,
            control_type_test_row,
            spellcheck_test_row,
            manual_evidence_row
        ]

        # Standardize and append all single-row test cases
        for row in single_row_tests:
            std_row = standardize_single_row(row)
            long_df = pd.concat([long_df, pd.DataFrame([std_row])], ignore_index=True)

        # Reorder the test "Types" as desired
        type_order = [
            "Predicted_DQ_Check_Nature",
            "Top3_DQ_Check_Instance_Names_Combined", 
            "Predicted_DQ_Check_Instance_Name",
            "FRASA_Criteria_Check",
            "Parsed_Column_Check",
            "Type_of_DQ_Check_Alignment",
            "Instance_Type_Alignment",
            "Version_Check",
            "Asset_Sheet_Fill_Check",
            "Data_Source_ID_Consistency_Check",
            "EDCC_Monitoring_Control_Tracking_ID_Check",
            "Risk_Instance_Name_Check",
            "Risk_Event_Level_3_Name_Check",
            "Control_Type_Level_3_Name_Check",
            "Spell_Check",
            "manual_evidence_check"
        ]

        unique_types = long_df["Type"].unique()
        for t in unique_types:
            if t not in type_order:
                type_order.append(t)

        type_order = [t for t in type_order if pd.notnull(t)]
        long_df["Type"] = long_df["Type"].fillna("Unknown")
        long_df["Type"] = pd.Categorical(long_df["Type"], categories=type_order, ordered=True)
        long_df = long_df.sort_values(["Type", "Check#"]).reset_index(drop=True)

        update_progress(100, "Analysis completed!")
        log_message("Analysis completed successfully!")
        return long_df
        
    except Exception as e:
        log_message(f"Error in analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    print("Enhanced DQ Analysis Core Module - Test Mode")
    print(f"Default instance file path: {DEFAULT_INSTANCE_FILE_PATH}")
    print(f"Default training data path: {DEFAULT_TRAINING_DATA_PATH}")
    print(f"Default model save path: {DEFAULT_MODEL_SAVE_PATH}")
    test_file = "test_file.xlsx"
    if os.path.exists(test_file):
        result = run_analysis(test_file)
        print(f"Analysis completed. Results shape: {result.shape}")
    else:
        print("No test file found. Module ready for import.")