import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

# Define the custom POS label set as requested
POS_LABELS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ", "PART", "NUM", "PUNCT"]

@st.cache_resource
def load_model():
    """
    Loads the multilingual pretrained model and tokenizer.
    Using 'xlm-roberta-base' as the base model and defining the classification head size.
    """
    model_name = "xlm-roberta-base"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with token classification head configured for our label set size
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(POS_LABELS))
    
    # Initialize the pipeline
    # We use aggregation_strategy="simple" to group subwords back into single words where possible
    pos_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return pos_pipeline

def predict_tags(text, pos_pipeline):
    """
    Runs the input text through the pipeline to get token classifications.
    """
    return pos_pipeline(text)

def main():
    # UI Elements
    st.title("Marathi POS Tagger")
    
    user_input = st.text_input("Enter Marathi sentence:")
    
    if st.button("Tag Sentence"):
        # Error handling for empty input
        if not user_input.strip():
            st.error("Please enter a valid Marathi sentence.")
        else:
            try:
                with st.spinner("Loading model and tagging sentence..."):
                    # Load model
                    pos_pipeline = load_model()
                    
                    # Run inference
                    predictions = predict_tags(user_input, pos_pipeline)
                    
                    # Display output clearly
                    st.success("Tagging Complete!")
                    st.markdown("### Results:")
                    
                    for pred in predictions:
                        word = pred.get('word', '')
                        
                        # Clean up subword tokenization artifacts (like ' ' in xlm-roberta)
                        if word.startswith(' '):
                            word = word[1:]
                            
                        # Handle the entity label
                        entity = pred.get('entity_group', pred.get('entity', 'LABEL_0'))
                        
                        # Map the model's output label (e.g., LABEL_1) to our custom POS_LABELS
                        try:
                            label_idx = int(entity.replace("LABEL_", ""))
                            pos_tag = POS_LABELS[label_idx]
                        except ValueError:
                            pos_tag = "UNKNOWN"
                            
                        # Only display valid, non-empty tokens
                        if word.strip():
                            st.write(f"**{word}** -> {pos_tag}")
                            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
