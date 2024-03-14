import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from tqdm.auto import tqdm
import argparse

def generate_impression(text, model, tokenizer):
    inputs = tokenizer(text.replace('\n', ' '),
                    padding="max_length",
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    outputs = model.generate(input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=512, 
                            num_beam_groups=1,
                            num_beams=4, 
                            do_sample=False,
                            diversity_penalty=0.0,
                            num_return_sequences=1, 
                            length_penalty=2.0,
                            no_repeat_ngram_size=3,
                            early_stopping=True
                            )
    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_str

def main(input_file, output_file, input_column, output_column):
    # Load the model and tokenizer
    finetuned_model = "xtie/PEGASUS-PET-impression"
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model) 
    model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model, ignore_mismatched_sizes=True).to("cuda").eval()

    # Load the DataFrame
    df = pd.read_csv(input_file)

    # Apply the function to generate impressions
    tqdm.pandas(desc="Generating Impressions")
    df[output_column] = df[input_column].progress_apply(lambda x: generate_impression(x, model, tokenizer))

    # Save or display the updated DataFrame
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Output saved to {output_file}")
    else:
        print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Radiologist Report Summaries")
    parser.add_argument("--input_column", type=str, default="radiologist_report", help="Name of the column containing the radiologist report")
    parser.add_argument("--output_column", type=str, default="rreport_summary", help="Name of the column to save the generated summary")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file containing 'radiologist_report' column")
    parser.add_argument("--output_file", type=str, help="Path to save the output CSV file. If not provided, output will be printed.")

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.input_column, args.output_column)
