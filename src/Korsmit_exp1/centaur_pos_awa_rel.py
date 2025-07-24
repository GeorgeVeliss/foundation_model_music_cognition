# NOTE: to be run on QMUL ssh servers, not locally

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

model_name = "marcelbinz/Llama-3.1-Centaur-70B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    pad_token_id=0,
    do_sample=True,
    temperature=1,
    max_new_tokens=14,
)

IDim_path_to_csv = "IDim0002.csv"

df = pd.read_csv(IDim_path_to_csv, sep=r"\s*,\s*", engine="python")
# Crucial: Strip whitespace from column names
df.columns = df.columns.str.strip()

# Map instrument abbreviations to full names
instrument_map = {
    "CTu": "Contrabass Tuba",
    "Tu": "Tuba",
    "CTb": "Contrabass Trombone",
    "BTb": "Bass Trombone",
    "TTb": "Tenor Trombone",
    "Ho": "Horn",
    "Tr": "Trumpet",
    "PTr": "Piccolo Trumpet",
    "CBa": "Contra Bassoon",
    "Ba": "Bassoon",
    "BCl": "Bass Clarinet",
    "ClB": "Clarinet Bb",
    "Ob": "Oboe",
    "EH": "English Horn",
    "AFl": "Alto Flute",
    "Fl": "Flute",
    "Ha": "Harp",
    "VC": "Cello",
    "Vl": "Violin",
    "Tmp": "Timpani",
    "Go": "Gong",
    "Ce": "Celesta",
    "Gsp": "Glockenspiel",
    "Xyl": "Xylophone",
    "Vib": "Vibraphone",
    "Cro": "Crotales",
}

# Ordinal map for octave values
ordinal_map = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
}

# Subset relevant columns
required_cols = ["instr", "octave"]
IDim_stimuli = df[required_cols].copy()

# Replace values
IDim_stimuli["instr"] = IDim_stimuli["instr"].map(instrument_map)
IDim_stimuli["octave"] = IDim_stimuli["octave"].astype(int).map(ordinal_map)


# Helper function for article and plural handling
def format_instrument_phrase(instr_name):
    plural_instruments = {"Timpani", "Crotales"}
    if instr_name in plural_instruments:
        return f"{instr_name}, playing in the {{octave}} octave"
    else:
        article = "an" if instr_name[0].lower() in "aeiou" else "a"
        return f"{article} {instr_name}, playing in the {{octave}} octave"


# Store responses
results_list = []

# Loop and collect responses
for stimulus, octave in zip(IDim_stimuli["instr"], IDim_stimuli["octave"]):
    instr_phrase = format_instrument_phrase(stimulus).format(octave=octave)

    prompt = (
        "You will hear a musical sound and rate how it makes you feel.\n"
        "Rate the sound on three continuous scales from 1 to 9:\n"
        "1. Negative to Positive\n"
        "2. Tired to Awake\n"
        "3. Tense to Relaxed\n"
        "Provide exactly three numbers, each with two decimals,\n"
        "formatted as: <<number1, number2, number3>>.\n\n"
        f"The sound is produced by {instr_phrase}.\n"
        "Your ratings are <<"
    )

    print(prompt)

    raw_response = pipe(prompt)[0]["generated_text"][len(prompt) :].strip()
    print(raw_response)

    # Parse the response (if it’s like "4, 7, 5")
    try:
        positive, awake, rest = [x.strip() for x in raw_response.split(",")]
        relaxed, rest = rest.split(">")
    except:
        awake = positive = relaxed = None

    # Store results
    results_list.append(
        {
            "stimulus": stimulus,
            "octave": octave,
            "prompt": prompt,
            "response_raw": raw_response,
            "awake": awake,
            "positive": positive,
            "relaxed": relaxed,
        }
    )

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    results_df.to_csv("centaur_responses_pos_awa_rel_240725.csv", index=False)

# Prompt for participant 0002 of IDim
# prompt = (
#     "You will hear musical sounds and rate how each makes you feel.\n"
#     "This is about the emotional quality you feel in response to the sound — not what the sound expresses.\n\n"
#     "Rate each sound on three scales from 1 to 9:\n"
#     "Negative to Positive\n"
#     "Tense to Relaxed\n"
#     "Tired to Awake\n\n"
#     f"The sound is produced by a trumpet playing the fourth octave. "
#     "Your scores are <<4.09, 4.02, 6.26>>.\n"
#     f"The sound is produced by a xylophone playing the sixth octave. "
#     "Your scores are <<7.19, 7.21, 2.40>>.\n"
#     f"The sound is produced by a horn playing the fifth octave. "
#     "Your scores are <<"
# )

# print(prompt)

# raw_response = pipe(prompt)[0]["generated_text"][len(prompt) :].strip()
# print(raw_response)
